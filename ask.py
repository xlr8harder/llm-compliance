import argparse
import datetime
import json
import os
import requests
import time
import tempfile # For atomic writing
# Removed shutil
import random   # For jitter

from pathlib import Path
from requests.exceptions import RequestException

# --- Constants ---
TRANSIENT_FAILURE_MARKER = None

# --- Helper Functions for Identifying Response Types ---

def is_permanent_api_error(resp):
    """Checks if the response structure indicates a permanent error reported by the API provider."""
    try:
        # Check for error within the 'choices' array (common pattern)
        if (isinstance(resp, dict) and
            "choices" in resp and
            isinstance(resp["choices"], list) and
            len(resp["choices"]) > 0 and
            isinstance(resp["choices"][0], dict) and
            "error" in resp["choices"][0] and
            resp["choices"][0]["error"] is not None): # Check error isn't explicitly null
            return True
        # Check for potential top-level error key (less common for chat completions but possible)
        if (isinstance(resp, dict) and
            "error" in resp and
            # Differentiate from our old transient error structure if needed, though that shouldn't be logged anymore
            # Ensure it looks like an API error object, not just our transient string
            isinstance(resp["error"], (dict, str)) and resp["error"] and
            "choices" not in resp): # Or if choices is empty/malformed alongside error
             return True
    except (TypeError, KeyError, IndexError):
        return False
    return False

def is_empty_content_response(resp):
    """
    Checks if a response object is structurally valid but has empty content.
    NOTE: This is now treated as a permanent error, potentially indicating censorship (e.g., Google).
    """
    try:
        if (isinstance(resp, dict) and
            "choices" in resp and
            isinstance(resp["choices"], list) and
            len(resp["choices"]) > 0 and
            isinstance(resp["choices"][0], dict) and
            # Make sure it's not already identified as a permanent API error
            "error" not in resp["choices"][0] and
            isinstance(resp["choices"][0].get("message"), dict) and
            resp["choices"][0]["message"].get("content") == ""):
            # It has the structure of success, but content is empty.
            return True
    except (TypeError, KeyError, IndexError):
        return False
    return False

# --- Loading and Cleanup Functions ---

def load_questions(file_path):
    """Load questions from a JSONL file."""
    questions = []
    path = Path(file_path)
    if not path.is_file():
        print(f"Error: Input questions file not found: {file_path}")
        return questions
    try:
        with path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    stripped_line = line.strip()
                    if stripped_line:
                        questions.append(json.loads(stripped_line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {i+1} in questions file {file_path}: {e}. Skipping line.")
    except Exception as e:
        print(f"Error reading questions file {file_path}: {e}")
    return questions

def load_existing_responses(output_file):
    """
    Load existing responses from output file.
    Adds IDs for ALL valid, parsable entries (successes and permanent errors)
    to the processed set for normal resume operation (skip what's logged).
    Returns:
        - processed_ids (set): IDs of questions with logged entries.
    """
    processed_ids = set()
    lines_read = 0
    parse_warnings = 0
    # Optional: Could still detect old/empty here just to print a warning, but not required for logic.

    if not output_file.exists():
        return processed_ids

    print(f"Loading existing responses from: {output_file}")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            lines_read = i + 1
            stripped_line = line.strip()
            if not stripped_line: continue
            try:
                entry = json.loads(stripped_line)
                question_id = entry.get("question_id")

                if question_id:
                    # Add ID for any valid, logged entry (success or permanent error)
                    processed_ids.add(question_id)
                else:
                    print(f"Warning: Line {lines_read} missing 'question_id'. Skipping.")
                    parse_warnings += 1

            except (json.JSONDecodeError) as e:
                print(f"Warning: Could not parse line {lines_read}: {str(e)}. Line: {stripped_line}")
                parse_warnings += 1
            except Exception as e:
                print(f"Warning: Error processing line {lines_read}: {type(e).__name__}: {str(e)}. Line: {stripped_line}")
                parse_warnings += 1

    print(f"Finished loading. Lines processed: {lines_read}")
    print(f"  - Found {len(processed_ids)} existing logged entries (Successes or Permanent Errors).")
    if parse_warnings > 0:
        print(f"  - Warnings during loading: {parse_warnings}")

    return processed_ids


def cleanup_permanent_errors(output_file):
    """
    Rewrites the output file, keeping ONLY successful responses.
    Removes entries with permanent API errors or empty content responses.
    Uses atomic replace for safety. Triggered by --force-retry-permanent-errors.
    """
    if not output_file.exists():
        print("Cleanup requested, but output file does not exist.")
        return False

    temp_file_path = None
    lines_read = 0
    lines_written = 0 # Successes kept
    api_errors_skipped = 0
    empty_skipped = 0
    corrupted_skipped = 0 # Now skipping corrupted lines during this specific cleanup

    print(f"\n--- Cleaning up permanent errors in: {output_file} ---")
    print("--- Keeping ONLY successful responses. ---")
    try:
        # Use NamedTemporaryFile for atomic operation
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=output_file.parent, suffix='.tmp') as temp_f:
            temp_file_path = Path(temp_f.name)

            with open(output_file, 'r', encoding='utf-8') as original_f:
                for i, line in enumerate(original_f):
                    lines_read = i + 1
                    stripped_line = line.strip()
                    if not stripped_line: continue

                    keep_line = False
                    try:
                        entry = json.loads(stripped_line)
                        response_data = entry.get("response")
                        question_id = entry.get("question_id") # Needed for logging skips

                        is_api_error = is_permanent_api_error(response_data)
                        is_empty = is_empty_content_response(response_data)

                        if not question_id:
                             print(f"Warning: Line {lines_read} missing ID during cleanup, skipping.")
                             corrupted_skipped += 1
                        elif is_api_error:
                            api_errors_skipped += 1
                            # print(f"  - Skipping line {lines_read} (ID: {question_id}): Permanent API Error.") # Verbose
                        elif is_empty:
                            empty_skipped += 1
                            # print(f"  - Skipping line {lines_read} (ID: {question_id}): Empty Content Error.") # Verbose
                        else:
                            # Assume it's a success if not a known permanent error
                            keep_line = True

                        if keep_line:
                            temp_f.write(line) # Write original line including newline
                            lines_written += 1

                    except (json.JSONDecodeError):
                         # Skip corrupted lines during this focused cleanup
                         print(f"Warning: Corrupted line {lines_read} encountered during cleanup, skipping.")
                         corrupted_skipped += 1

        # --- Atomic Rename ---
        if lines_read > 0:
             print(f"\nCleanup processed {lines_read} lines.")
             print(f"  - Kept {lines_written} successful responses.")
             print(f"  - Skipped {api_errors_skipped} permanent API errors.")
             print(f"  - Skipped {empty_skipped} empty content responses.")
             print(f"  - Skipped {corrupted_skipped} corrupted/unparseable lines.")
             print(f"Attempting to atomically replace original file with cleaned version (only successes)...")
             os.replace(temp_file_path, output_file) # Atomic rename
             print(f"Successfully replaced {output_file}.")
             temp_file_path = None # Prevent deletion in finally
             print("--- Cleanup Finished Successfully ---")
             return True
        else: # lines_read == 0
             print("Original file was empty. No cleanup performed.")
             if temp_file_path and temp_file_path.exists():
                 temp_file_path.unlink()
             return False

    except Exception as e:
        print(f"!!! ERROR during cleanup process: {type(e).__name__}: {e} !!!")
        import traceback
        traceback.print_exc()
        return False # Indicate cleanup failed
    finally:
        # Ensure temporary file is deleted if rename didn't happen
        if temp_file_path and temp_file_path.exists():
            print(f"Cleaning up temporary file: {temp_file_path}")
            try:
                temp_file_path.unlink()
            except OSError as unlink_e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {unlink_e}")

# --- API Interaction ---

def retry_with_backoff(func, max_retries=8, initial_delay=1):
    """
    Retry a function with exponential backoff. Handles transient errors ONLY.
    Empty content or API errors are returned directly by func() and passed through.
    Returns the result of func() on first success or permanent error.
    Returns TRANSIENT_FAILURE_MARKER (None) if all retries fail due to transient exceptions.
    """
    retries = 0
    last_exception = None
    while retries < max_retries:
        try:
            # Call the function (e.g., make_request)
            # It should return the response directly if it's a success,
            # a permanent API error, or an empty content response.
            # It should raise an exception only for *transient* issues.
            result = func()
            return result # Return immediately on any non-exception result

        # --- Handle exceptions that trigger retries ---
        except (RequestException, ValueError, json.JSONDecodeError, KeyError) as e:
            # These exceptions indicate transient problems (network, server 5xx/429, bad JSON decode, etc.)
            last_exception = e
            print(f"Attempt {retries + 1}/{max_retries} failed (transient): {type(e).__name__}: {str(e)}")
            retries += 1
            if retries == max_retries:
                print(f"Max retries reached. Final transient error: {type(last_exception).__name__}: {str(last_exception)}.")
                return TRANSIENT_FAILURE_MARKER

            # Calculate delay with exponential backoff and jitter
            base_delay = initial_delay * (2 ** (retries -1))
            jitter = base_delay * 0.1
            actual_delay = min(base_delay + random.uniform(-jitter, jitter), 60)
            actual_delay = max(actual_delay, 0)

            print(f"Retrying in {actual_delay:.2f} seconds...")
            time.sleep(actual_delay)

    # Fallback
    print("Exited retry loop unexpectedly.")
    return TRANSIENT_FAILURE_MARKER


def ask_question(question, model, api_key):
    """
    Send a question to the API endpoint. Returns JSON response on success/permanent error,
    or TRANSIENT_FAILURE_MARKER (None) if retries exhausted for transient issues.
    """
    if model.startswith("accounts/fireworks"):
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        api_type = "Fireworks"
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        api_type = "OpenRouter"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    if api_type == "OpenRouter":
         headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERRER", "http://localhost")
         headers["X-Title"] = os.getenv("OPENROUTER_TITLE", "AI Question Answering Script")

    data = {
        "model": model,
        "messages": [ {"role": "user", "content": question} ]
    }

    # Inner function passed to retry_with_backoff
    def make_request():
        response = requests.post(
            url=url, headers=headers, data=json.dumps(data), timeout=120
        )

        # --- Check for TRANSIENT HTTP errors first ---
        if response.status_code != 200:
             error_details = f"Status Code: {response.status_code}"
             try:
                 resp_json = response.json()
                 error_msg = resp_json.get('error', {}).get('message', str(resp_json))
                 error_details += f", Body: {error_msg}"
             except json.JSONDecodeError:
                 error_details += f", Body: {response.text[:500]}"

             # Raise ONLY for retryable status codes (Server errors, Rate limits)
             if response.status_code >= 500 or response.status_code == 429:
                 raise RequestException(f"Retryable server/rate limit error: {error_details}")
             else:
                 # Treat other client errors (4xx) as potentially permanent *at this level*.
                 # We return an error structure instead of raising, so retry_with_backoff doesn't retry.
                 print(f"Client error encountered: {error_details}. Treating as permanent failure for this request.")
                 # Construct a synthetic permanent error response (or return the actual if parsable)
                 try:
                     return response.json() # Return actual error response if possible
                 except json.JSONDecodeError:
                      return {"error": {"message": f"Permanent Client Error: {error_details}", "code": response.status_code}}


        # --- If status 200, parse and check content ---
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            # JSON decode error on 200 OK is likely transient server issue
            raise ValueError(f"Failed to decode JSON on 200 OK: {e}. Response: {response.text[:500]}")

        if not isinstance(response_data, dict):
             # Malformed success response -> treat as transient
             raise ValueError(f"Unexpected response format (not dict): {type(response_data)}")

        # Check for permanent API error or empty content WITHIN the valid 200 response
        # These are considered permanent failures and should be returned directly.
        if is_permanent_api_error(response_data):
            print("Detected permanent API error in response content.")
            # No action needed, just return it
        elif is_empty_content_response(response_data):
             # NOTE: Treating empty content as a permanent error (potential censorship)
             print("Detected empty content in response. Treating as permanent failure.")
             # No action needed, just return it

        # Otherwise, it's a valid success response
        return response_data

    # Call retry_with_backoff - it will only retry on exceptions raised by make_request
    return retry_with_backoff(make_request)


# --- Main Processing Logic ---

def process_questions(questions_file, model, api_key, force_restart=False, force_retry_permanent=False):
    """Process questions, save valid responses, handle resumes and cleanup."""
    output_dir = Path("responses")
    output_dir.mkdir(exist_ok=True)

    questions_filename = Path(questions_file).stem
    safe_model_name = model.replace('/', '_')
    output_file = output_dir / f"{questions_filename}_{safe_model_name}.jsonl"

    questions = load_questions(questions_file)
    if not questions: return
    total_questions = len(questions)
    print(f"Loaded {total_questions} questions from {questions_file}")

    # --- Handle Force Retry Permanent Errors ---
    if force_retry_permanent:
        print("\n--force-retry-permanent-errors flag detected.")
        if output_file.exists():
            cleanup_successful = cleanup_permanent_errors(output_file)
            if not cleanup_successful:
                print("ERROR: Cleanup of permanent errors failed. Exiting to prevent unexpected behavior.")
                return # Exit if cleanup fails when explicitly requested
            print("Permanent error cleanup finished.")
        else:
            print("Output file does not exist, no permanent errors to clean.")
        # After cleanup, we proceed as if starting fresh or resuming from only successes.

    # --- Load existing state ---
    # This now loads IDs of all entries remaining after potential cleanup
    processed_ids = load_existing_responses(output_file)

    if not force_retry_permanent: # Only print resume message if not forced retry
        if output_file.exists():
            print(f"Resuming. {len(processed_ids)} completed entries (Successes/Permanent Errors) found.")
        else:
             print(f"No existing output file found at '{output_file}'. Starting fresh.")

    if force_restart and not force_retry_permanent: # force_restart implies removing everything
        print(f"Force restart requested. Removing existing output file '{output_file}'.")
        if output_file.exists():
            try:
                output_file.unlink()
                processed_ids = set() # Reset processed IDs
                print("Existing file removed.")
            except OSError as e:
                print(f"Error removing file {output_file}: {e}. Exiting.")
                return
        else:
            print("Force restart requested, but no file existed.")
            processed_ids = set()


    # --- Main processing loop ---
    print(f"\nProcessing remaining questions...")
    print(f"Saving ALL responses (Successes and Permanent Errors) to: {output_file}")

    questions_processed_this_run = 0
    questions_skipped_this_run = 0
    questions_failed_transiently_this_run = 0
    questions_logged_this_run = 0

    start_time = time.time()

    for i, question_data in enumerate(questions, 1):
        question_id = question_data.get('id')
        if not question_id:
            print(f"Warning: Question index {i-1} missing 'id'. Skipping.")
            continue

        # Skip questions that have an existing log entry (unless --force-retry-permanent was used and cleaned it)
        if question_id in processed_ids:
            questions_skipped_this_run += 1
            continue

        completed_count = len(processed_ids) # Recalculate inside loop? No, load once.
        # This progress calculation assumes we process linearly after skipping
        questions_yet_to_process = total_questions - completed_count
        progress_percent = (completed_count / total_questions) * 100

        print(f"\n--- Processing Q {i}/{total_questions} (ID: {question_id}) with {model} --- ({questions_yet_to_process} remaining, {progress_percent:.1f}%)")
        questions_processed_this_run += 1

        response = ask_question(
            question_data.get('question', '[Question Missing]'), model, api_key
        )

        # --- Handle response ---
        if response is TRANSIENT_FAILURE_MARKER:
            questions_failed_transiently_this_run += 1
            print(f"-> Q ID {question_id}: Transient failure after retries. Not logged.")
        else:
            # Log success or ANY permanent error (API error, empty content, etc.)
            output_entry = {
                "question_id": question_id,
                "category": question_data.get('category', 'N/A'),
                "question": question_data.get('question', 'N/A'),
                "model": model,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "response": response
            }
            if 'domain' in question_data: output_entry['domain'] = question_data['domain']

            log_status = "Unknown"
            if is_permanent_api_error(response):
                log_status = "Permanent Error (API)"
            elif is_empty_content_response(response):
                log_status = "Permanent Error (Empty Content)"
            # Basic check for success (has choices, first choice has message content, not error)
            elif isinstance(response, dict) and "choices" in response and len(response.get("choices",[])) > 0 and response["choices"][0].get("message",{}).get("content"):
                 log_status = "Success"
            else:
                 log_status = "Permanent Error (Other/Malformed)" # Catch-all for unexpected permanent issues


            try:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
                # Add ID to processed set only after successful write
                processed_ids.add(question_id)
                questions_logged_this_run += 1
                # --- Updated Log Message ---
                print(f"-> Q ID {question_id}: Logged response ({log_status}).")
                # --- End Updated Log Message ---
            except Exception as e:
                 print(f"CRITICAL ERROR writing entry for Q ID {question_id}: {e}")


    # --- Final Summary ---
    end_time = time.time()
    duration = end_time - start_time
    print("-" * 30)
    print(f"Finished processing {questions_file}.")
    print(f"  Total questions: {total_questions}")
    print(f"  Attempted this run: {questions_processed_this_run}")
    print(f"  Skipped (already logged): {questions_skipped_this_run}")
    print(f"  Logged this run (Success/PermError): {questions_logged_this_run}")
    print(f"  Failed transiently this run (will retry): {questions_failed_transiently_this_run}")
    print(f"  Processing duration: {duration:.2f} seconds")

    # Reload final count for accuracy
    final_processed_ids = load_existing_responses(output_file)
    print(f"  Total logged entries in file now: {len(final_processed_ids)}/{total_questions}")
    print(f"  Output file: {output_file}")
    print("-" * 30)


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description='Ask questions via API, with retries for transient errors. Logs successes and permanent errors (API errors, empty content). Can force retry permanent errors.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('model', help='Model identifier (e.g., "openai/gpt-4o")')
    parser.add_argument('questions_files', nargs='+', help='One or more JSONL question files.')
    parser.add_argument('--force-restart', action='store_true',
                        help='Delete existing output file and start fresh (overrides --force-retry-permanent-errors).')
    parser.add_argument('--force-retry-permanent-errors', '--frpe', action='store_true',
                        help='Remove all logged permanent errors (API errors, empty content) from the output file before starting, forcing them to be retried.')

    args = parser.parse_args()

    # --- Argument Sanity Check ---
    if args.force_restart and args.force_retry_permanent_errors:
        print("Warning: --force-restart overrides --force-retry-permanent-errors. Proceeding with full restart.")
        args.force_retry_permanent_errors = False # Ensure only force_restart logic runs

    is_fireworks = args.model.startswith('accounts/fireworks')
    env_var = 'FIREWORKS_API_KEY' if is_fireworks else 'OPENROUTER_API_KEY'
    api_key = os.getenv(env_var)

    if not api_key:
        model_type = "Fireworks" if is_fireworks else "OpenRouter/Compatible"
        print(f"Error: {env_var} environment variable must be set for {model_type} models.")
        exit(1)

    print(f"Using API Key from env var: {env_var}")
    if not is_fireworks:
         print(f"OpenRouter Referrer: {os.getenv('OPENROUTER_REFERRER', 'http://localhost')}")
         print(f"OpenRouter Title: {os.getenv('OPENROUTER_TITLE', 'AI Question Answering Script')}")

    total_start_time = time.time()
    files_processed_count = 0
    files_failed_count = 0

    for questions_file in args.questions_files:
        print(f"\n=== Starting processing for file: {questions_file} ===")
        file_start_time = time.time()
        if not Path(questions_file).is_file():
             print(f"Error: Input file not found: {questions_file}. Skipping.")
             files_failed_count += 1
             continue
        try:
            # Pass the new argument to process_questions
            process_questions(
                questions_file,
                args.model,
                api_key,
                args.force_restart,
                args.force_retry_permanent_errors # Pass the flag
            )
            files_processed_count += 1
        except Exception as e:
            files_failed_count += 1
            print(f"\n!!! CRITICAL UNEXPECTED ERROR processing {questions_file} !!!")
            print(f"Error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Attempting to continue...")
        file_end_time = time.time()
        print(f"=== Finished processing file: {questions_file} (Duration: {file_end_time - file_start_time:.2f} seconds) ===")

    total_end_time = time.time()
    print("\n" + "="*40)
    print("Overall Summary:")
    print(f"  Total files attempted: {len(args.questions_files)}")
    print(f"  Files processed: {files_processed_count}")
    print(f"  Files failed/skipped: {files_failed_count}")
    print(f"  Total execution time: {total_end_time - total_start_time:.2f} seconds")
    print("="*40)

if __name__ == "__main__":
    main()
