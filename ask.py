import argparse
import datetime
import json
import os
import requests
import time
import tempfile # For atomic writing
import shutil   # For renaming across filesystems if needed
import random   # <--- Moved import random to the top

from pathlib import Path
from requests.exceptions import RequestException

# --- Constants ---
# Marker returned by retry_with_backoff on complete transient failure
TRANSIENT_FAILURE_MARKER = None

# --- Helper for Identifying Old Transient Errors ---

def is_old_transient_retry_error(resp):
    """
    Checks if a response object matches the format of a previously logged
    transient error (from the old retry_with_backoff).
    """
    return (
        isinstance(resp, dict) and
        "error" in resp and
        "choices" not in resp and
        "id" not in resp and
        isinstance(resp.get("error"), str) and
        resp["error"].startswith("After ") # Loosely check prefix
    )

# --- Loading and Cleanup Functions ---

def load_questions(file_path):
    """Load questions from a JSONL file."""
    questions = []
    path = Path(file_path)
    if not path.is_file():
        print(f"Error: Input questions file not found: {file_path}")
        return questions # Return empty list if file doesn't exist
    try:
        with path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    stripped_line = line.strip()
                    if stripped_line: # Avoid errors from empty lines
                        questions.append(json.loads(stripped_line))
                    # else: # Optional: Log empty lines in input
                    #      print(f"Warning: Empty line {i+1} in questions file {file_path}. Skipping.")
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {i+1} in questions file {file_path}: {e}. Skipping line.")
    except Exception as e:
        print(f"Error reading questions file {file_path}: {e}")
    return questions

def load_existing_responses(output_file):
    """
    Load existing responses from output file.
    Adds question IDs to the processed set if their corresponding entry exists
    (implicitly meaning it's a success or a permanent API error).
    Also detects if any *old-style* transient errors are present.
    """
    processed_ids = set()
    transient_errors_detected = False
    lines_read = 0
    parse_warnings = 0

    if not output_file.exists():
        return processed_ids, transient_errors_detected # Return empty set and False

    print(f"Loading existing responses from: {output_file}")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            lines_read = i + 1
            stripped_line = line.strip()
            if not stripped_line: # Skip empty lines silently in output
                continue
            try:
                entry = json.loads(stripped_line)
                question_id = entry.get("question_id")
                response_data = entry.get("response")

                if not question_id:
                    print(f"Warning: Line {lines_read} missing 'question_id' in output file. Skipping line.")
                    parse_warnings += 1
                    continue

                # Check if this line represents an OLD transient error
                if is_old_transient_retry_error(response_data):
                    transient_errors_detected = True
                    # Do NOT add its ID to processed_ids, it needs to be retried (and cleaned up)
                    print(f"  - ID {question_id}: Old transient error detected (will be removed if cleanup runs).")
                else:
                    # Assume any other valid entry is permanent (success or API error)
                    processed_ids.add(question_id)
                    # --- Optional detailed logging (currently commented out) ---
                    # is_permanent_api_error = False
                    # if isinstance(response_data, dict) and "choices" in response_data:
                    #      if isinstance(response_data["choices"], list) and len(response_data["choices"]) > 0:
                    #          first_choice = response_data["choices"][0]
                    #          if isinstance(first_choice, dict) and "error" in first_choice:
                    #              is_permanent_api_error = True
                    #              print(f"  - ID {question_id}: Permanent API error found, will NOT retry.")
                    #
                    # if not is_permanent_api_error and not is_old_transient_retry_error(response_data) :
                    #      print(f"  - ID {question_id}: Success or unknown structure found, will NOT retry.")


            except (json.JSONDecodeError) as e:
                print(f"Warning: Could not parse line {lines_read} in output file: {str(e)}. Line: {stripped_line}")
                parse_warnings += 1
            except Exception as e: # Catch other potential issues
                print(f"Warning: Error processing line {lines_read} in output file: {type(e).__name__}: {str(e)}. Line: {stripped_line}")
                parse_warnings += 1

    print(f"Finished loading. Lines processed: {lines_read}")
    print(f"  - IDs marked as completed (found in file, non-transient): {len(processed_ids)}")
    if transient_errors_detected:
        print("  - *** Old-style transient errors detected in the file. Cleanup will be attempted. ***")
    if parse_warnings > 0:
        print(f"  - Warnings during loading: {parse_warnings}")

    return processed_ids, transient_errors_detected


def cleanup_transient_errors(output_file):
    """
    Reads the output file, writes a new version excluding old transient errors
    to a temporary file, and then atomically replaces the original file.
    """
    if not output_file.exists():
        print("Cleanup requested, but output file does not exist.")
        return False # Nothing to cleanup

    temp_file_path = None
    lines_read = 0
    lines_written = 0
    transient_lines_skipped = 0
    corrupted_lines_kept = 0

    print(f"\n--- Starting cleanup of transient errors in: {output_file} ---")
    try:
        # Create a temporary file in the same directory for atomic rename
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=output_file.parent, suffix='.tmp') as temp_f:
            temp_file_path = Path(temp_f.name)
            # print(f"Writing cleaned content to temporary file: {temp_file_path}") # Verbose logging

            with open(output_file, 'r', encoding='utf-8') as original_f:
                for i, line in enumerate(original_f):
                    lines_read = i + 1
                    stripped_line = line.strip()
                    if not stripped_line: # Skip empty lines in original
                        continue

                    try:
                        entry = json.loads(stripped_line)
                        response_data = entry.get("response")

                        if is_old_transient_retry_error(response_data):
                            # Skip writing this line
                            transient_lines_skipped += 1
                        else:
                            # Write valid non-transient lines to the temp file
                            temp_f.write(line) # Write original line including newline
                            lines_written += 1
                    except (json.JSONDecodeError):
                         # If line is corrupted, preserve it to avoid data loss
                         print(f"Warning: Corrupted line {lines_read} encountered during cleanup, preserving it.")
                         temp_f.write(line) # Write original line including newline
                         lines_written += 1
                         corrupted_lines_kept += 1

        # --- Atomic Rename ---
        # Only proceed if we actually wrote something to the temp file or if the original only had transient errors
        if lines_written > 0 :
             print(f"Cleanup processed {lines_read} lines. Wrote {lines_written} lines to temp file (skipped {transient_lines_skipped} transient errors, kept {corrupted_lines_kept} corrupted lines).")
             print(f"Attempting to atomically replace original file...")
             os.replace(temp_file_path, output_file)
             print(f"Successfully replaced {output_file} with cleaned version.")
             temp_file_path = None # Prevent deletion in finally block
             print("--- Cleanup Finished Successfully ---")
             return True
        elif lines_read > 0 and transient_lines_skipped == lines_read:
            # The file only contained transient errors, so the clean file is empty
            print(f"Cleanup processed {lines_read} lines. Original file only contained transient errors.")
            print(f"Attempting to replace original file with empty file...")
            os.replace(temp_file_path, output_file) # Replace with the empty temp file
            print(f"Successfully replaced {output_file} with empty cleaned version.")
            temp_file_path = None
            print("--- Cleanup Finished Successfully ---")
            return True
        else: # lines_read == 0
             print("Original file was empty. No cleanup needed.")
             if temp_file_path and temp_file_path.exists():
                 temp_file_path.unlink() # Should not happen, but be safe
             return False


    except Exception as e:
        print(f"!!! ERROR during cleanup process: {type(e).__name__}: {e} !!!")
        import traceback
        traceback.print_exc()
        return False # Indicate cleanup failed
    finally:
        # Ensure temporary file is deleted if something went wrong before rename
        if temp_file_path and temp_file_path.exists():
            print(f"Cleaning up temporary file due to error or incomplete process: {temp_file_path}")
            try:
                temp_file_path.unlink()
            except OSError as unlink_e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {unlink_e}")


# --- API Interaction ---

def retry_with_backoff(func, max_retries=8, initial_delay=1):
    """
    Retry a function with exponential backoff.
    Returns the result of func() on success or permanent API error.
    Returns TRANSIENT_FAILURE_MARKER (None) if all retries fail due to transient exceptions.
    """
    retries = 0
    last_exception = None
    while retries < max_retries:
        try:
            result = func()
            # Check for permanent API errors returned within the response structure
            if isinstance(result, dict) and "choices" in result and isinstance(result["choices"], list):
                 if len(result["choices"]) > 0 and isinstance(result["choices"][0], dict) and "error" in result["choices"][0]:
                     error_content = result['choices'][0]['error']
                     print(f"API returned an error within 'choices': {error_content}. Treating as permanent failure.")
                     return result # Return the error structure, it's permanent

            # Check for empty response content which might be considered a failure
            # Example: if choices[0].message.content is None or empty
            # if isinstance(result, dict) and "choices" in result and isinstance(result["choices"], list):
            #      if len(result["choices"]) > 0 and isinstance(result["choices"][0].get("message"), dict):
            #          if not result["choices"][0]["message"].get("content"):
            #              print("Received empty content in response message. Treating as transient failure.")
            #              raise ValueError("Received empty content in choices") # Force retry

            # If no permanent errors detected, return the result (could be success)
            return result

        except (RequestException, ValueError, json.JSONDecodeError, KeyError) as e:
            last_exception = e # Store the last exception for potential logging
            print(f"Attempt {retries + 1}/{max_retries} failed: {type(e).__name__}: {str(e)}")
            retries += 1
            if retries == max_retries:
                print(f"Max retries reached. Final error: {type(last_exception).__name__}: {str(last_exception)}. Transient failure for this attempt.")
                return TRANSIENT_FAILURE_MARKER # Return the marker for transient failure

            # Calculate delay with exponential backoff and jitter
            base_delay = initial_delay * (2 ** (retries -1)) # Exponential backoff (start from initial_delay)
            jitter = base_delay * 0.1 # Add up to 10% jitter
            actual_delay = min(base_delay + random.uniform(-jitter, jitter), 60) # Cap delay at 60s
            actual_delay = max(actual_delay, 0) # Ensure delay is non-negative

            print(f"Retrying in {actual_delay:.2f} seconds...")
            time.sleep(actual_delay)

    # Fallback, should ideally not be reached if max_retries > 0
    print("Exited retry loop unexpectedly.")
    return TRANSIENT_FAILURE_MARKER


def ask_question(question, model, api_key):
    """
    Send a question to the appropriate API endpoint based on model name.
    Returns the JSON response on success or permanent API error.
    Returns TRANSIENT_FAILURE_MARKER (None) if retries are exhausted due to transient issues.
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
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }

    def make_request():
        response = requests.post(
            url=url,
            headers=headers,
            data=json.dumps(data),
            timeout=120
        )

        if response.status_code != 200:
             error_details = f"Status Code: {response.status_code}"
             try:
                 resp_json = response.json()
                 error_msg = resp_json.get('error', {}).get('message', str(resp_json))
                 error_details += f", Body: {error_msg}"
             except json.JSONDecodeError:
                 error_details += f", Body: {response.text[:500]}"

             # Raise exceptions for retryable status codes (Server errors, Rate limits)
             if response.status_code >= 500 or response.status_code == 429:
                 raise RequestException(f"Retryable error: {error_details}")
             else: # Treat other client errors (4xx) as retryable for now
                 print(f"Client error encountered: {error_details}. Treating as transient for retry purposes.")
                 raise RequestException(f"Client error: {error_details}")

        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON on 200 OK: {e}. Response text: {response.text[:500]}")

        if not isinstance(response_data, dict):
             raise ValueError(f"Unexpected response format (not a dict): {type(response_data)}")

        return response_data

    return retry_with_backoff(make_request)


# --- Main Processing Logic ---

def process_questions(questions_file, model, api_key, force_restart=False):
    """Process all questions in a file and save responses. Resume if output file exists."""
    output_dir = Path("responses")
    output_dir.mkdir(exist_ok=True)

    questions_filename = Path(questions_file).stem
    # --- Reverted Filename Logic ---
    # Keep hyphens, periods, etc. Only replace '/' which is common and invalid in filenames.
    safe_model_name = model.replace('/', '_')
    output_file = output_dir / f"{questions_filename}_{safe_model_name}.jsonl"
    # --- End Reverted Filename Logic ---

    questions = load_questions(questions_file)
    if not questions:
        print(f"No valid questions loaded from {questions_file}. Aborting processing for this file.")
        return
    total_questions = len(questions)
    print(f"Loaded {total_questions} questions from {questions_file}")


    processed_ids = set()
    transient_errors_detected_in_file = False

    # --- Load existing state and check for cleanup ---
    if output_file.exists() and not force_restart:
        print(f"Checking existing output file: {output_file}") # Explicitly state checking
        processed_ids, transient_errors_detected_in_file = load_existing_responses(output_file)
        if transient_errors_detected_in_file:
            cleanup_successful = cleanup_transient_errors(output_file)
            if cleanup_successful:
                print("Reloading processed IDs from the cleaned file...")
                processed_ids, _ = load_existing_responses(output_file)
            else:
                print("WARNING: Cleanup of transient errors failed. Proceeding with potentially inaccurate resume state.")

        print(f"Resuming. {len(processed_ids)} completed questions found.")

    elif force_restart and output_file.exists():
        print(f"Force restart requested. Removing existing output file '{output_file}'.")
        try:
            output_file.unlink()
            processed_ids = set()
            print("Existing file removed.")
        except OSError as e:
            print(f"Error removing existing file {output_file}: {e}. Exiting to prevent data issues.")
            return
    else:
         # Handle case where file doesn't exist OR force_restart was true but file didn't exist
        if not output_file.exists():
             print(f"No existing output file found at '{output_file}'. Starting fresh.")
        elif force_restart:
             print(f"Force restart requested, but no file existed at '{output_file}'. Starting fresh.")
        processed_ids = set()


    # --- Main processing loop ---
    print(f"\nProcessing remaining questions...")
    print(f"Saving successes/permanent errors to: {output_file}")

    questions_processed_this_run = 0
    questions_skipped_this_run = 0
    questions_failed_transiently_this_run = 0
    questions_logged_this_run = 0

    start_time = time.time()

    for i, question_data in enumerate(questions, 1):
        question_id = question_data.get('id')
        if not question_id:
            print(f"Warning: Question at index {i-1} missing 'id'. Skipping.")
            continue

        if question_id in processed_ids:
            questions_skipped_this_run += 1
            continue

        # Calculate progress based on remaining questions
        completed_count = len(processed_ids)
        total_to_process = total_questions - completed_count
        current_attempt_num = questions_processed_this_run + 1
        progress_percent = (current_attempt_num / total_to_process) * 100 if total_to_process > 0 else 100

        print(f"\n--- Processing question {i}/{total_questions} (ID: {question_id}) --- (Attempt {current_attempt_num}/{total_to_process} of remaining, {progress_percent:.1f}%)")
        questions_processed_this_run += 1

        response = ask_question(
            question_data.get('question', '[Question Missing]'),
            model,
            api_key
        )

        # --- Handle response ---
        if response is TRANSIENT_FAILURE_MARKER:
            questions_failed_transiently_this_run += 1
            print(f"-> Question ID {question_id} failed transiently after all retries. Not logged.")
        else:
            # Success or permanent API error
            output_entry = {
                "question_id": question_id,
                "category": question_data.get('category', 'N/A'),
                "question": question_data.get('question', 'N/A'),
                "model": model,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "response": response
            }
            if 'domain' in question_data:
                output_entry['domain'] = question_data['domain']

            try:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
                processed_ids.add(question_id) # Update in-memory set
                questions_logged_this_run += 1
                print(f"-> Logged response (Success or Permanent Error) for ID {question_id}.")
            except Exception as e:
                 print(f"CRITICAL ERROR: Failed to write entry for question ID {question_id} to {output_file}: {e}")
                 # If write fails, don't add to processed_ids in memory for this run,
                 # allowing it to be retried next time.

        # time.sleep(0.1) # Optional delay

    # --- Final Summary ---
    end_time = time.time()
    duration = end_time - start_time
    print("-" * 30)
    print(f"Finished processing {questions_file}.")
    print(f"Total questions in file: {total_questions}")
    print(f"Questions attempted this run: {questions_processed_this_run}")
    print(f"Questions skipped (already completed): {questions_skipped_this_run}")
    print(f"Questions logged this run (Success or Permanent Error): {questions_logged_this_run}")
    print(f"Questions failed transiently this run (will retry next time): {questions_failed_transiently_this_run}")
    print(f"Processing duration: {duration:.2f} seconds")

    final_processed_ids, _ = load_existing_responses(output_file)
    print(f"Total completed entries in output file now: {len(final_processed_ids)}/{total_questions}")
    print(f"Output file location: {output_file}")
    print("-" * 30)


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description='Ask questions to AI models via OpenRouter or Fireworks API, with retries and resumability. Logs only successes and permanent errors. Cleans up old transient errors automatically.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('model',
                        help='Model identifier (e.g., "openai/gpt-4o", "accounts/fireworks/models/llama-v3-70b-instruct")')
    parser.add_argument('questions_files', nargs='+',
                        help='One or more JSONL files containing questions (keys: "id", "question", optional "category", "domain").')
    parser.add_argument('--force-restart', action='store_true',
                        help='Delete existing output file for the given model/file combo and start fresh.')

    args = parser.parse_args()

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
    files_processed = 0
    files_failed = 0

    for questions_file in args.questions_files:
        print(f"\n=== Starting processing for file: {questions_file} ===")
        file_start_time = time.time()
        if not Path(questions_file).is_file():
             print(f"Error: Input file not found: {questions_file}. Skipping.")
             files_failed += 1
             continue
        try:
            process_questions(
                questions_file,
                args.model,
                api_key,
                args.force_restart
            )
            files_processed += 1
        except Exception as e:
            files_failed += 1
            print(f"\n!!! An critical unexpected error occurred outside the main loop for {questions_file} !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            import traceback
            print("\n--- Traceback ---")
            traceback.print_exc()
            print("--- End Traceback ---\n")
            print("Attempting to continue with the next file if any...")
        file_end_time = time.time()
        print(f"=== Finished processing for file: {questions_file} (Duration: {file_end_time - file_start_time:.2f} seconds) ===")

    total_end_time = time.time()
    print("\n" + "="*40)
    print("Overall Summary:")
    print(f"Total files attempted: {len(args.questions_files)}")
    print(f"Files processed successfully: {files_processed}")
    print(f"Files failed or skipped due to errors: {files_failed}")
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds")
    print("="*40)


if __name__ == "__main__":
    main()
