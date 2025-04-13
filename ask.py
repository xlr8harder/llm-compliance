import argparse
import datetime
import json
import os
import requests
import time
import tempfile
import random
import threading 
import traceback
import concurrent.futures
from pathlib import Path
from requests.exceptions import RequestException

# --- Constants ---
TRANSIENT_FAILURE_MARKER = None
# Add a counter for overall progress reporting
PROGRESS_COUNTER = 0
PROGRESS_LOCK = threading.Lock()

# --- Helper Functions for Identifying Response Types ---
def is_permanent_api_error(resp):
    """Checks if the response structure indicates a permanent error reported by the API provider."""
    try:
        if (isinstance(resp, dict) and
            "choices" in resp and
            isinstance(resp["choices"], list) and
            len(resp["choices"]) > 0 and
            isinstance(resp["choices"][0], dict) and
            "error" in resp["choices"][0] and
            resp["choices"][0]["error"] is not None):
            return True
        if (isinstance(resp, dict) and
            "error" in resp and
            isinstance(resp["error"], (dict, str)) and resp["error"] and
            "choices" not in resp):
             return True
    except (TypeError, KeyError, IndexError):
        return False
    return False

def is_empty_content_response(resp):
    """Checks if a response object is structurally valid but has empty content."""
    try:
        if (isinstance(resp, dict) and
            "choices" in resp and
            isinstance(resp["choices"], list) and
            len(resp["choices"]) > 0 and
            isinstance(resp["choices"][0], dict) and
            "error" not in resp["choices"][0] and
            isinstance(resp["choices"][0].get("message"), dict) and
            resp["choices"][0]["message"].get("content") == ""):
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
        return questions, False
    try:
        valid_count = 0
        skipped_count = 0
        with path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    stripped_line = line.strip()
                    if stripped_line:
                        entry = json.loads(stripped_line)
                        if 'id' in entry and 'question' in entry:
                            questions.append(entry)
                            valid_count += 1
                        else:
                            print(f"Warning: Skipping line {i+1} in questions file {file_path}: missing 'id' or 'question'.")
                            skipped_count += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {i+1} in questions file {file_path}: {e}. Skipping line.")
                    skipped_count += 1
        print(f"Loaded {valid_count} valid questions from {file_path}.")
        if skipped_count > 0: print(f"Skipped {skipped_count} invalid/malformed lines.")
        return questions, True
    except Exception as e:
        print(f"Error reading questions file {file_path}: {e}")
        return questions, False

def load_existing_responses(output_file):
    """Load existing response IDs from output file."""
    processed_ids = set()
    lines_read, parse_warnings = 0, 0
    if not output_file.exists(): return processed_ids
    print(f"Loading existing responses from: {output_file}")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            lines_read = i + 1
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                entry = json.loads(stripped_line)
                question_id = entry.get("question_id")
                if question_id:
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
    """Rewrites the output file, keeping ONLY successful responses."""
    if not output_file.exists():
        print("Cleanup requested, but output file does not exist.")
        return False
    temp_file_path = None
    lines_read, lines_written, api_errors_skipped, empty_skipped, corrupted_skipped = 0, 0, 0, 0, 0
    print(f"\n--- Cleaning up permanent errors in: {output_file} ---")
    print("--- Keeping ONLY successful responses. ---")
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=output_file.parent, suffix='.tmp') as temp_f:
            temp_file_path = Path(temp_f.name)
            with open(output_file, 'r', encoding='utf-8') as original_f:
                for i, line in enumerate(original_f):
                    lines_read = i + 1
                    stripped_line = line.strip()

                    if not stripped_line:
                        continue
                    keep_line = False
                    try:
                        entry = json.loads(stripped_line)
                        response_data, question_id = entry.get("response"), entry.get("question_id")
                        is_api_error, is_empty = is_permanent_api_error(response_data), is_empty_content_response(response_data)
                        if not question_id:
                            corrupted_skipped += 1
                            print(f"Warning: Line {lines_read} missing ID during cleanup, skipping.")
                        elif is_api_error:
                            api_errors_skipped += 1
                        elif is_empty:
                            empty_skipped += 1
                        else:
                            keep_line = True
                        if keep_line:
                            temp_f.write(line)
                            lines_written += 1
                    except (json.JSONDecodeError):
                        corrupted_skipped += 1
                        print(f"Warning: Corrupted line {lines_read} encountered during cleanup, skipping.")
        if lines_read > 0:
             print(f"\nCleanup processed {lines_read} lines.")
             print(f"  - Kept {lines_written} successful responses.")
             print(f"  - Skipped {api_errors_skipped} permanent API errors.")
             print(f"  - Skipped {empty_skipped} empty content responses.")
             print(f"  - Skipped {corrupted_skipped} corrupted/unparseable lines.")
             print(f"Attempting to atomically replace original file...")
             os.replace(temp_file_path, output_file)
             print(f"Successfully replaced {output_file}.")
             temp_file_path = None
             print("--- Cleanup Finished Successfully ---")
             return True
        else:
             print("Original file was empty. No cleanup performed.")
             if temp_file_path and temp_file_path.exists():
                 temp_file_path.unlink()
             return False
    except Exception as e:
        print(f"!!! ERROR during cleanup process: {type(e).__name__}: {e} !!!")
        traceback.print_exc()
        return False
    finally:
        if temp_file_path and temp_file_path.exists():
            print(f"Cleaning up temporary file: {temp_file_path}")
            try:
                temp_file_path.unlink()
            except OSError as unlink_e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {unlink_e}")

# --- API Interaction ---
def retry_with_backoff(func, max_retries=8, initial_delay=1):
    """Retry a function with exponential backoff for transient errors."""
    retries = 0
    last_exception = None
    while retries < max_retries:
        try:
            result = func()
            return result
        except (RequestException, ValueError, json.JSONDecodeError, KeyError) as e:
            last_exception = e
            retries += 1
            if retries == max_retries:
                # Log clearly when giving up due to transient errors
                print(f"\n[Thread-{threading.get_ident()}] Max retries ({max_retries}) reached. Final transient error: {type(last_exception).__name__}: {str(last_exception)}.")
                return TRANSIENT_FAILURE_MARKER
            base_delay = initial_delay * (2 ** (retries -1))
            jitter = base_delay * 0.1
            actual_delay = min(base_delay + random.uniform(-jitter, jitter), 60)
            actual_delay = max(actual_delay, 0)
            time.sleep(actual_delay)
    print(f"\n[Thread-{threading.get_ident()}] Exited retry loop unexpectedly.") # Should not happen
    return TRANSIENT_FAILURE_MARKER

def ask_question(question, model, api_key):
    """Send a question to the API endpoint. Handles retries internally."""
    if model.startswith("accounts/fireworks"):
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        api_type = "Fireworks"
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        api_type = "OpenRouter"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if api_type == "OpenRouter":
         headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERRER", "http://localhost")
         headers["X-Title"] = os.getenv("OPENROUTER_TITLE", "AI Question Answering Script")
    data = {"model": model, "messages": [ {"role": "user", "content": question} ]}
    def make_request():
        response = requests.post(url=url, headers=headers, data=json.dumps(data), timeout=120)
        if response.status_code != 200:
             error_details = f"Status Code: {response.status_code}"
             try:
                 resp_json = response.json()
                 error_msg = resp_json.get('error', {}).get('message', str(resp_json))
                 error_details += f", Body: {error_msg}"
             except json.JSONDecodeError:
                 error_details += f", Body: {response.text[:500]}"
             if response.status_code >= 500 or response.status_code == 429:
                 raise RequestException(f"Retryable server/rate limit error: {error_details}")
             else:
                 # Print non-retryable client errors clearly
                 print(f"\n[Thread-{threading.get_ident()}] Client error: {error_details}. Treating as permanent failure.")
                 try:
                     return response.json()
                 except json.JSONDecodeError:
                     return {"error": {"message": f"Permanent Client Error: {error_details}", "code": response.status_code}}
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON on 200 OK: {e}. Response: {response.text[:500]}")
        if not isinstance(response_data, dict):
            raise ValueError(f"Unexpected response format (not dict): {type(response_data)}")
        # Don't log success/empty/perm error detection here, let caller handle
        return response_data
    return retry_with_backoff(make_request) # Calls the retry logic

# --- Worker Function ---
def process_single_question_worker(question_data, model, api_key):
    """Worker function to process a single question."""
    response = ask_question(question_data.get('question', '[Missing Question]'), model, api_key)
    return question_data, response

# --- Main Processing Logic ---
def process_questions(questions_file, model, api_key, num_workers, force_restart=False, force_retry_permanent=False):
    """Process questions in parallel, save responses, handle resumes and cleanup."""
    global PROGRESS_COUNTER # Use global counter
    PROGRESS_COUNTER = 0 # Reset counter for each file

    output_dir = Path("responses")
    output_dir.mkdir(exist_ok=True)
    questions_filename = Path(questions_file).stem
    safe_model_name = model.replace('/', '_')
    output_file = output_dir / f"{questions_filename}_{safe_model_name}.jsonl"

    questions, loaded_ok = load_questions(questions_file)
    if not loaded_ok:
        return
    total_questions = len(questions)
    print(f"Loaded {total_questions} questions from {questions_file}")

    if force_retry_permanent:
        print("\n--force-retry-permanent-errors flag detected.")
        if output_file.exists():
            cleanup_successful = cleanup_permanent_errors(output_file)
            if not cleanup_successful:
                print("ERROR: Cleanup failed. Exiting.")
                return
            print("Permanent error cleanup finished.")
        else:
            print("Output file does not exist, no permanent errors to clean.")
    processed_ids = load_existing_responses(output_file)
    if not force_retry_permanent:
        if output_file.exists():
            print(f"Resuming. {len(processed_ids)} completed entries found.")
        else:
            print(f"No existing output file found at '{output_file}'. Starting fresh.")
    if force_restart and not force_retry_permanent:
        print(f"Force restart requested. Removing existing output file '{output_file}'.")
        if output_file.exists():
            try:
                output_file.unlink()
                processed_ids = set()
                print("Existing file removed.")
            except OSError as e:
                print(f"Error removing file {output_file}: {e}. Exiting.")
                return
        else:
            print("Force restart requested, but no file existed.")
            processed_ids = set()

    questions_to_process = []
    for q_data in questions:
        q_id = q_data.get('id')
        if not q_id:
            print(f"Warning: Question missing 'id', skipping: {q_data.get('question', '')[:50]}...")
            continue
        if q_id not in processed_ids:
            questions_to_process.append(q_data)

    num_to_process = len(questions_to_process)
    num_skipped = total_questions - num_to_process
    print(f"\nTotal questions: {total_questions}")
    print(f"Already processed: {num_skipped}")
    print(f"Questions to process this run: {num_to_process}")
    if num_to_process == 0:
        print("No questions remaining to process.")
        return

    print(f"\nProcessing {num_to_process} questions using {num_workers} workers...")
    print(f"Saving ALL responses (Successes and Permanent Errors) to: {output_file}")
    output_lock = threading.Lock()
    questions_processed_this_run, questions_logged_this_run, questions_failed_transiently_this_run = 0, 0, 0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_question_worker, q_data, model, api_key)
                   for q_data in questions_to_process]

        for future in concurrent.futures.as_completed(futures):
            try:
                question_data, response = future.result()
                question_id = question_data.get('id')
                questions_processed_this_run += 1 # Count completed task

                # --- Handle response ---
                if response is TRANSIENT_FAILURE_MARKER:
                    questions_failed_transiently_this_run += 1
                else:
                    output_entry = {
                        "question_id": question_id,
                        "category": question_data.get('category'),
                        "question": question_data.get('question'),
                        "model": model,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "response": response
                    }
                    if 'domain' in question_data:
                        output_entry['domain'] = question_data['domain']

                    # --- Determine log status ---
                    log_status = "Unknown"
                    if is_permanent_api_error(response):
                        log_status = "PermError(API)"
                    elif is_empty_content_response(response):
                        log_status = "PermError(Empty)"
                    elif isinstance(response, dict) and "choices" in response and len(response.get("choices",[])) > 0 and response["choices"][0].get("message",{}).get("content"):
                        log_status = "Success"
                    else:
                        log_status = "PermError(Other)"

                    # --- Thread-safe writing to output file ---
                    try:
                        with output_lock: # Keep the lock for explicit safety
                            with open(output_file, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
                        questions_logged_this_run += 1

                        # --- Simple Progress Indicator ---
                        with PROGRESS_LOCK: # Lock the counter update/read
                             PROGRESS_COUNTER += 1
                             # Print progress every N items or based on percentage
                             if PROGRESS_COUNTER % 10 == 0 or PROGRESS_COUNTER == num_to_process:
                                 elapsed = time.time() - start_time
                                 rate = PROGRESS_COUNTER / elapsed if elapsed > 0 else 0
                                 print(f"  Progress: {PROGRESS_COUNTER}/{num_to_process} ({PROGRESS_COUNTER/num_to_process:.1%}) completed. Rate: {rate:.2f} Q/s.")

                    except Exception as e:
                         print(f"\nCRITICAL ERROR writing entry for Q ID {question_id}: {e}")

            except Exception as e:
                 print(f"\nCRITICAL ERROR processing future result: {type(e).__name__}: {e}")
                 traceback.print_exc()

    # --- Final Summary ---
    end_time = time.time()
    duration = end_time - start_time
    print("-" * 30)
    print(f"Finished processing {questions_file}.")
    print(f"  Total questions in file: {total_questions}")
    print(f"  Tasks submitted this run: {num_to_process}")
    print(f"  Tasks completed by workers: {questions_processed_this_run}")
    print(f"  Responses logged this run (Success/PermError): {questions_logged_this_run}")
    print(f"  Tasks failed transiently this run (will retry): {questions_failed_transiently_this_run}")
    print(f"  Processing duration: {duration:.2f} seconds")
    final_processed_ids = load_existing_responses(output_file)
    print(f"  Total logged entries in file now: {len(final_processed_ids)} / {total_questions}")
    print(f"  Output file: {output_file}")
    print("-" * 30)

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description='Ask questions via API in parallel, with retries for transient errors. Logs successes and permanent errors.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', help='Model identifier (e.g., "openai/gpt-4o")')
    parser.add_argument('questions_files', nargs='+', help='One or more JSONL question files (must contain "id" and "question" fields).')
    parser.add_argument('-w', '--workers', type=int, default=1, help='Number of parallel worker threads for API requests.')
    parser.add_argument('--force-restart', action='store_true', help='Delete existing output file and start fresh (overrides --force-retry-permanent-errors).')
    parser.add_argument('--force-retry-permanent-errors', '--frpe', action='store_true', help='Remove logged permanent errors from the output file before starting, forcing retry.')
    args = parser.parse_args()

    if args.workers < 1:
        print("Error: Number of workers must be at least 1.")
        exit(1)
    if args.force_restart and args.force_retry_permanent_errors:
        print("Warning: --force-restart overrides --force-retry-permanent-errors. Proceeding with full restart.")
        args.force_retry_permanent_errors = False

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
    print(f"Using {args.workers} parallel worker(s).")

    total_start_time = time.time()
    files_processed_count, files_failed_count = 0, 0
    for questions_file in args.questions_files:
        print(f"\n=== Starting processing for file: {questions_file} ===")
        file_start_time = time.time()
        if not Path(questions_file).is_file():
            print(f"Error: Input file not found: {questions_file}. Skipping.")
            files_failed_count += 1
            continue
        try:
            process_questions(questions_file, args.model, api_key, args.workers, args.force_restart, args.force_retry_permanent_errors)
            files_processed_count += 1
        except Exception as e:
            files_failed_count += 1
            print(f"\n!!! CRITICAL UNEXPECTED ERROR processing {questions_file} !!!")
            print(f"Error: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            print("Attempting to continue...")
        file_end_time = time.time()
        print(f"=== Finished processing file: {questions_file} (Duration: {file_end_time - file_start_time:.2f} seconds) ===")

    total_end_time = time.time()
    print("\n" + "="*40)
    print("Overall Summary:")
    print(f"  Total files attempted: {len(args.questions_files)}")
    print(f"  Files processed successfully: {files_processed_count}")
    print(f"  Files failed/skipped: {files_failed_count}")
    print(f"  Total execution time: {total_end_time - total_start_time:.2f} seconds")
    print("="*40)

if __name__ == "__main__":
    main()
