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
import sys
from pathlib import Path
from requests.exceptions import RequestException

# --- Constants ---
TRANSIENT_FAILURE_MARKER = object() # Use a unique object marker
# Add a counter for overall progress reporting
PROGRESS_COUNTER = 0
PROGRESS_LOCK = threading.Lock()
# Define valid provider names
VALID_PROVIDERS = ["openai", "openrouter", "fireworks", "chutes", "google"]

# --- API Endpoint Constants ---
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"
CHUTES_API_BASE = "https://llm.chutes.ai/v1"
GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# --- Helper Functions for Identifying Response Types ---
def is_permanent_api_error(resp):
    """Checks if the response structure indicates a permanent error reported by the API provider."""
    # This function relies on the standardized 'response' field structure.
    # Google responses are adapted to this format before this check.
    try:
        # Case 1: Error within choices (e.g., content filter, safety block)
        if (isinstance(resp, dict) and
            "choices" in resp and
            isinstance(resp["choices"], list) and
            len(resp["choices"]) > 0 and
            isinstance(resp["choices"][0], dict) and
            "error" in resp["choices"][0] and
             resp["choices"][0]["error"] is not None):
            return True
        # Case 2: Top-level error (common for auth, rate limits AFTER retries, invalid request)
        if (isinstance(resp, dict) and
            "error" in resp and
            isinstance(resp["error"], (dict, str)) and resp["error"] and
            "choices" not in resp):
             return True
    except (TypeError, KeyError, IndexError):
        # Malformed response might indicate an error, but hard to classify as permanent *here*.
        # Let retry logic handle network/decode issues. If it persists, it gets logged eventually.
        return False
    return False

def is_empty_content_response(resp):
    """Checks if a response object is structurally valid but has empty content."""
    # This function relies on the standardized 'response' field structure.
    try:
        if (isinstance(resp, dict) and
            "choices" in resp and
            isinstance(resp["choices"], list) and
            len(resp["choices"]) > 0 and
            isinstance(resp["choices"][0], dict) and
            resp["choices"][0].get("error") is None and # Explicitly check error is None or missing
            isinstance(resp["choices"][0].get("message"), dict) and
            resp["choices"][0]["message"].get("content") == ""): # Check for strictly empty string
            return True
    except (TypeError, KeyError, IndexError):
        return False
    return False

# --- Loading and Cleanup Functions ---
# load_questions, load_existing_responses, cleanup_permanent_errors remain unchanged
# (Assuming they are correct as provided in the prompt)
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
        # Use NamedTemporaryFile correctly within a 'with' statement
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=output_file.parent, suffix='.tmp') as temp_f:
            temp_file_path = Path(temp_f.name) # Get the path
            with open(output_file, 'r', encoding='utf-8') as original_f:
                for i, line in enumerate(original_f):
                    lines_read = i + 1
                    stripped_line = line.strip()

                    if not stripped_line:
                        continue # Skip empty lines

                    keep_line = False
                    try:
                        entry = json.loads(stripped_line)
                        # Check for required fields before processing
                        # The 'response' field should hold the standardized structure
                        response_data = entry.get("response")
                        question_id = entry.get("question_id")

                        if not question_id:
                            corrupted_skipped += 1
                            print(f"Warning: Line {lines_read} missing 'question_id' during cleanup, skipping.")
                            continue # Skip this line

                        if response_data is None: # Handle case where 'response' key is missing
                            corrupted_skipped +=1
                            print(f"Warning: Line {lines_read} (QID: {question_id}) missing 'response' data during cleanup, skipping.")
                            continue # Skip this line

                        # Now check the content of the response using the helper functions
                        # which expect the standardized format
                        is_api_error = is_permanent_api_error(response_data)
                        is_empty = is_empty_content_response(response_data)

                        if is_api_error:
                            api_errors_skipped += 1
                        elif is_empty:
                            empty_skipped += 1
                        else:
                             # Add a check for basic success structure before keeping
                             # This still checks the standardized 'response' field
                             if (isinstance(response_data, dict) and
                                 "choices" in response_data and
                                 isinstance(response_data["choices"], list) and
                                 len(response_data["choices"]) > 0 and
                                 isinstance(response_data["choices"][0].get("message"), dict) and
                                 response_data["choices"][0]["message"].get("content") is not None): # Allow non-empty content
                                 keep_line = True
                             else:
                                 # Treat responses that aren't errors/empty but lack valid content as corrupted/unusable
                                 corrupted_skipped += 1
                                 # print(f"Warning: Line {lines_read} (QID: {question_id}) lacks valid content structure, skipping.")


                        if keep_line:
                            temp_f.write(line) # Write the original line
                            lines_written += 1

                    except (json.JSONDecodeError):
                        corrupted_skipped += 1
                        print(f"Warning: Corrupted line {lines_read} encountered during cleanup (JSONDecodeError), skipping.")
                    except Exception as e_inner: # Catch other potential errors processing a line
                         corrupted_skipped += 1
                         print(f"Warning: Error processing line {lines_read} during cleanup: {type(e_inner).__name__}. Skipping.")


        # --- Post-processing and replacement ---
        if lines_read > 0:
             print(f"\nCleanup processed {lines_read} lines.")
             print(f"  - Kept {lines_written} successful responses.")
             print(f"  - Skipped {api_errors_skipped} permanent API errors.")
             print(f"  - Skipped {empty_skipped} empty content responses.")
             print(f"  - Skipped {corrupted_skipped} corrupted/unparseable/invalid structure lines.")

             if lines_written > 0 or api_errors_skipped > 0 or empty_skipped > 0 or corrupted_skipped > 0 : # Only replace if changes were made or file non-empty
                 print(f"Attempting to atomically replace original file...")
                 try:
                     # Ensure temp file is closed before replacing (it is by end of 'with')
                     os.replace(temp_file_path, output_file)
                     print(f"Successfully replaced {output_file}.")
                     temp_file_path = None # Avoid deletion in finally block
                     print("--- Cleanup Finished Successfully ---")
                     return True
                 except OSError as e_replace:
                      print(f"!!! ERROR replacing file {output_file} with {temp_file_path}: {e_replace} !!!")
                      return False # Indicate failure
             else:
                 print("No valid lines kept and no errors/empty found to remove. Original file likely empty or contained only corrupted lines.")
                 # No need to replace if nothing changed
                 print("--- Cleanup Finished (No Changes Made) ---")
                 return True


        else: # Original file was empty or only had blank lines
             print("Original file was empty or contained no processable lines. No cleanup performed.")
             # No need to replace
             return True # Not an error state

    except Exception as e:
        print(f"!!! ERROR during cleanup process: {type(e).__name__}: {e} !!!")
        traceback.print_exc()
        return False
    finally:
        # Clean up temp file if replacement failed or wasn't attempted
        if temp_file_path and temp_file_path.exists():
            print(f"Cleaning up temporary file: {temp_file_path}")
            try:
                temp_file_path.unlink()
            except OSError as unlink_e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {unlink_e}")


# --- API Interaction ---

def retry_with_backoff(func, question_id, api_provider_name, max_retries=8, initial_delay=1):
    """Retry a function with exponential backoff for transient errors."""
    retries = 0
    last_exception = None
    while retries < max_retries:
        try:
            result = func()
            # Check for non-200 responses that indicate permanent client errors
            if isinstance(result, dict):
                 status_code = result.get("status_code") # Check if make_request added this
                 # Non-retryable status codes (client errors, excluding 429)
                 if status_code and 400 <= status_code < 500 and status_code != 429:
                    error_info = result.get("error", {})
                    # Specific non-retryable error types can be checked here if providers use them
                    # e.g., OpenAI's 'invalid_request_error'
                    if isinstance(error_info, dict) and error_info.get("type") == "invalid_request_error":
                         print(f"\n[Thread-{threading.get_ident()}] QID {question_id}: {api_provider_name} API: Permanent client error ({error_info.get('type', 'N/A')}). Not retrying.")
                         return {"error": error_info}
                    # Google might indicate permanent errors differently, but 4xx codes are a good general indicator

                    # Generic non-retryable client error
                    print(f"\n[Thread-{threading.get_ident()}] QID {question_id}: {api_provider_name} API: Non-retryable client error ({status_code}). Treating as permanent failure.")
                    return {"error": result.get("error_detail", f"Permanent Client Error {status_code}")}

            return result # Return successful response or result triggering exception below

        # Catch exceptions that indicate a *potentially* transient issue
        except (RequestException, ValueError, json.JSONDecodeError, KeyError) as e:
            last_exception = e
            retries += 1
            error_type_name = type(last_exception).__name__
            error_details = str(last_exception)

            # Add more details if it's a RequestException with a response
            if isinstance(last_exception, RequestException) and last_exception.response is not None:
                status = last_exception.response.status_code
                body_snippet = last_exception.response.text[:200].replace('\n', ' ') # Limit snippet size
                error_details = f"Status: {status}, Body: {body_snippet}..."

            if retries == max_retries:
                print(f"\n[Thread-{threading.get_ident()}] QID {question_id}: {api_provider_name} API: Max retries ({max_retries}) reached. Final transient error: {error_type_name}: {error_details}.")
                return TRANSIENT_FAILURE_MARKER

            # Log retry attempt with more details
            print(f"\n[Thread-{threading.get_ident()}] QID {question_id}: {api_provider_name} API: Attempt {retries}/{max_retries} failed ({error_type_name}: {error_details}). Retrying...")

            base_delay = initial_delay * (2 ** (retries - 1))
            jitter = base_delay * 0.1
            actual_delay = max(0, min(base_delay + random.uniform(-jitter, jitter), 60)) # Cap max delay
            time.sleep(actual_delay)

    # Should not be reached if max_retries > 0
    print(f"\n[Thread-{threading.get_ident()}] QID {question_id}: {api_provider_name} API: Exited retry loop unexpectedly.")
    return TRANSIENT_FAILURE_MARKER


def adapt_google_response(google_response, model_id_used):
    """Adapts a Google Gemini API response to the standard format and includes raw data."""
    adapted = {
        "id": None, # Google doesn't provide a standard ID for the completion itself
        "object": "chat.completion", # Assume standard object type
        "model": model_id_used, # Model used for the API call
        "choices": [],
        "usage": {},
        "_raw_provider_response": google_response # Store the original response
    }

    try:
        # Usage mapping
        if 'usageMetadata' in google_response:
            um = google_response['usageMetadata']
            adapted['usage'] = {
                'prompt_tokens': um.get('promptTokenCount'),
                'completion_tokens': um.get('candidatesTokenCount'),
                'total_tokens': um.get('totalTokenCount')
            }

        # Try to get the specific model version if available
        if 'usageMetadata' in google_response and 'modelVersion' in google_response['usageMetadata']: # Correction: modelVersion seems to be peer to usageMetadata
             pass # Keep original model_id_used if not found, maybe log warning?
        elif 'modelVersion' in google_response: # Check top level as seen in example
             adapted['model'] = google_response['modelVersion']


        # Process candidates (usually just one)
        if 'candidates' in google_response and isinstance(google_response['candidates'], list) and len(google_response['candidates']) > 0:
            candidate = google_response['candidates'][0]
            choice = {
                "index": 0,
                "message": {"role": "assistant", "content": ""}, # Initialize message
                "finish_reason": None,
                "logprobs": None # Not typically available in basic Google response
            }

            # Extract finish reason
            finish_reason_raw = candidate.get('finishReason')
            if finish_reason_raw:
                # Map Google reasons to standard ones (simple mapping for now)
                reason_map = {
                    "STOP": "stop",
                    "MAX_TOKENS": "length",
                    "SAFETY": "content_filter", # Treat safety as content filter
                    "RECITATION": "content_filter", # Treat recitation as content filter
                    "OTHER": "error",
                    "UNSPECIFIED": "error"
                }
                choice['finish_reason'] = reason_map.get(finish_reason_raw, finish_reason_raw.lower()) # Default to lowercase raw reason

            # Extract content
            if 'content' in candidate and 'parts' in candidate['content'] and isinstance(candidate['content']['parts'], list) and len(candidate['content']['parts']) > 0:
                # Assuming the first part contains the main text response
                choice['message']['content'] = candidate['content']['parts'][0].get('text', '')

            # Handle safety/error finish reasons by adding an error object to the choice
            if choice['finish_reason'] in ["content_filter", "error"]:
                choice["error"] = {
                    "message": f"Response stopped due to: {finish_reason_raw}",
                    "type": choice['finish_reason'], # Use the mapped reason as type
                    "code": finish_reason_raw # Store the original Google reason code
                }
                # Keep partial content if available when error/filtered
                # choice['message']['content'] = choice['message']['content'] # Content is already extracted

            adapted['choices'].append(choice)
        # Handle cases where the prompt itself was blocked (no candidates)
        elif 'promptFeedback' in google_response and google_response['promptFeedback'].get('blockReason'):
             block_reason = google_response['promptFeedback']['blockReason']
             safety_ratings = google_response['promptFeedback'].get('safetyRatings', [])
             choice = {
                 "index": 0,
                 "message": {"role": "assistant", "content": ""}, # No content generated
                 "finish_reason": "content_filter", # Treat prompt block as content filter
                 "error": {
                     "message": f"Prompt blocked due to: {block_reason}",
                     "type": "content_filter",
                     "code": block_reason,
                     "param": "prompt", # Indicate the issue was with the prompt
                     "safety_ratings": safety_ratings
                 }
             }
             adapted['choices'].append(choice)
        # Handle cases where response is missing candidates or content otherwise (treat as error)
        elif not adapted['choices']:
             adapted['error'] = {'message': 'Invalid response structure: Missing candidates or promptFeedback', 'type': 'invalid_response'}

    except (KeyError, IndexError, TypeError, AttributeError) as e:
        print(f"Error adapting Google response: {e}")
        traceback.print_exc()
        adapted['error'] = {'message': f'Failed to adapt Google response structure: {e}', 'type': 'adaptation_error'}
        # Ensure choices list exists even if adaptation failed partially
        if 'choices' not in adapted: adapted['choices']=[]


    return adapted


def ask_question(question, question_id, model_id, api_target, api_key):
    """Send a question to the specified API endpoint. Handles retries internally."""
    url = ""
    headers = {"Content-Type": "application/json"} # Default header
    data = {} # Payload varies by provider
    api_provider_name = api_target # Use lowercase target name directly

    # --- Configure API details based on target ---
    if api_target == "openai":
        url = f"{OPENAI_API_BASE}/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        data = {"model": model_id, "messages": [{"role": "user", "content": question}]}

    elif api_target == "openrouter":
        url = f"{OPENROUTER_API_BASE}/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERRER", "http://localhost")
        headers["X-Title"] = os.getenv("OPENROUTER_TITLE", "AI Question Answering Script")
        data = {"model": model_id, "messages": [{"role": "user", "content": question}]}

    elif api_target == "fireworks":
        url = f"{FIREWORKS_API_BASE}/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        data = {"model": model_id, "messages": [{"role": "user", "content": question}]}

    elif api_target == "chutes":
        url = f"{CHUTES_API_BASE}/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        data = {"model": model_id, "messages": [{"role": "user", "content": question}]}

    elif api_target == "google":
        # Note: Google uses API key as query param, not Auth header
        url = f"{GOOGLE_API_BASE}/models/{model_id}:generateContent?key={api_key}"
        # Google uses a different payload structure
        data = {"contents": [{"parts": [{"text": question}]}]}
        # Example: Add safety settings if needed
        # data["safetySettings"] = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        # ]

    else:
        print(f"ERROR: Unknown api_target '{api_target}' in ask_question.")
        return {"error": {"message": f"Internal configuration error: Unknown API target '{api_target}'", "code": "config_error"}}

    # --- Define the request function for retry logic ---
    def make_request():
        response = requests.post(url=url, headers=headers, data=json.dumps(data), timeout=180)

        # --- Handle non-200 responses ---
        if response.status_code != 200:
            status_code = response.status_code
            error_details = f"Status Code: {status_code}"
            error_json_payload = None # Holds the structured error from the response body
            raw_response_body = response.text # Keep raw body for details

            try:
                resp_json = response.json()
                # Google errors are often under an 'error' key
                if 'error' in resp_json and isinstance(resp_json['error'], dict):
                    error_json_payload = resp_json['error']
                    error_msg = error_json_payload.get('message', str(resp_json))
                else: # Fallback for other providers or unexpected Google errors
                    error_json_payload = resp_json.get('error', resp_json) # Take 'error' if present, else whole body
                    error_msg = str(error_json_payload.get('message', error_json_payload)) if isinstance(error_json_payload, dict) else str(error_json_payload)

                error_details += f", Body: {error_msg}"
                # Store raw json if parsing succeeded
                raw_response_body = resp_json

            except json.JSONDecodeError:
                error_details += f", Body (non-JSON): {raw_response_body[:500]}"
            except Exception as parse_exc:
                 error_details += f", Body (error parsing): {type(parse_exc).__name__} - {raw_response_body[:500]}"

            # --- Classify error for retry/failure ---
            # Raise RequestException for retryable errors (5xx, 429)
            # Include the response object in the exception for better logging in retry_with_backoff
            if status_code >= 500 or status_code == 429:
                raise RequestException(f"Retryable {api_provider_name} server/rate limit error: {error_details}", response=response)

            # Specific non-retryable client errors (e.g., OpenAI invalid request)
            elif isinstance(error_json_payload, dict) and error_json_payload.get("type") == "invalid_request_error":
                 # Pass the structured error and details up
                 return {"status_code": status_code, "error": error_json_payload, "error_detail": error_details, "_raw_provider_response": raw_response_body}
            # Other client errors (4xx except 429) - treat as permanent
            else:
                 print(f"\n[Thread-{threading.get_ident()}] QID {question_id}: {api_provider_name} API Client error: {error_details}. Treating as permanent failure.")
                 # Ensure we return a standard error structure
                 if not isinstance(error_json_payload, (dict, str)):
                      error_json_payload = {"message": f"Permanent Client Error: {error_details}", "code": status_code}
                 return {"status_code": status_code, "error": error_json_payload, "error_detail": error_details, "_raw_provider_response": raw_response_body}

        # --- Handle successful 200 OK response ---
        try:
            raw_response_data = response.json() # Get the raw JSON data
            if not isinstance(raw_response_data, dict):
                raise ValueError(f"Unexpected {api_provider_name} response format (not dict): {type(raw_response_data)}")

            # Adapt response based on provider
            if api_target == 'google':
                adapted_response = adapt_google_response(raw_response_data, model_id)
                # Add internal metadata needed by calling functions
                adapted_response["_provider_used"] = api_provider_name
                adapted_response["_model_used_for_api"] = model_id
                return adapted_response
            else:
                # For other providers, assume standard format and just add metadata
                raw_response_data["_provider_used"] = api_provider_name
                raw_response_data["_model_used_for_api"] = model_id
                # Store raw response for consistency, though it's the same as the main body here
                raw_response_data["_raw_provider_response"] = raw_response_data.copy()
                return raw_response_data

        except json.JSONDecodeError as e:
            # Raise ValueError for retry logic to catch potentially transient decode issue
            raise ValueError(f"Failed to decode {api_provider_name} JSON on 200 OK: {e}. Response: {response.text[:500]}")
        except Exception as e:
             # Raise ValueError for retry logic for other adaptation errors
             raise ValueError(f"Error processing/adapting successful {api_provider_name} response: {type(e).__name__}: {e}")


    # --- Execute with retry logic ---
    return retry_with_backoff(make_request, question_id, api_provider_name)

# --- Worker Function ---
def process_single_question_worker(question_data, model_id, api_target, api_key):
    """Worker function to process a single question using the determined API target."""
    question_id = question_data.get('id', '[Missing ID]') # Get question ID here
    response = ask_question(
        question_data.get('question', '[Missing Question]'),
        question_id, # Pass question_id to ask_question
        model_id,
        api_target,
        api_key
        )
    # Response dict may contain standard fields AND '_raw_provider_response'
    return question_data, response

# --- Main Processing Logic ---
def process_questions(questions_file, model_id, canonical_name, api_target, api_key, num_workers, force_restart=False, force_retry_permanent=False):
    """Process questions in parallel, save responses, handle resumes and cleanup."""
    global PROGRESS_COUNTER
    PROGRESS_COUNTER = 0 # Reset counter for each file

    output_dir = Path("responses")
    output_dir.mkdir(exist_ok=True)
    questions_filename = Path(questions_file).stem
    # Use canonical name if provided for filename, else use the model_id
    output_model_name = canonical_name if canonical_name else model_id
    safe_output_model_name = output_model_name.replace('/', '_').replace(':', '-')
    output_file = output_dir / f"{questions_filename}_{safe_output_model_name}.jsonl"

    questions, loaded_ok = load_questions(questions_file)
    if not loaded_ok:
        return
    total_questions = len(questions)

    # --- Resume / Restart Logic ---
    if force_retry_permanent:
        print("\n--force-retry-permanent-errors flag detected.")
        if output_file.exists():
            cleanup_successful = cleanup_permanent_errors(output_file)
            if not cleanup_successful:
                print("ERROR: Cleanup failed. Cannot proceed with retrying permanent errors.")
                return
            print("Permanent error cleanup finished.")
        else:
            print("Output file does not exist, no permanent errors to clean.")

    processed_ids = set() # Reset processed IDs after potential cleanup
    if output_file.exists():
         if not force_restart:
              processed_ids = load_existing_responses(output_file)
              print(f"Resuming. Found {len(processed_ids)} completed entries in '{output_file}'.")
         elif force_retry_permanent:
              processed_ids = load_existing_responses(output_file)
              print(f"Found {len(processed_ids)} entries remaining after permanent error cleanup in '{output_file}'.")
         else: # force_restart is True and not force_retry_permanent
             print(f"Force restart requested. Removing existing output file '{output_file}'.")
             try:
                 output_file.unlink()
                 processed_ids = set() # Start fresh
                 print("Existing file removed.")
             except OSError as e:
                 print(f"Error removing file {output_file}: {e}. Exiting file processing.")
                 return
    else:
         print(f"No existing output file found at '{output_file}'. Starting fresh.")


    # --- Filter Questions ---
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
    print(f"\nTotal questions in file: {total_questions}")
    print(f"Already processed (or skipped in prior runs): {num_skipped}")
    print(f"Questions to process this run: {num_to_process}")
    if num_to_process == 0:
        print("No questions remaining to process for this file.")
        return

    # --- Start Processing ---
    model_name_to_log = canonical_name if canonical_name else model_id
    print(f"\nProcessing {num_to_process} questions for model '{model_name_to_log}' using API target '{api_target.upper()}' with {num_workers} workers...")
    print(f"Saving ALL responses (Successes and Permanent Errors) to: {output_file}")
    output_lock = threading.Lock()
    questions_processed_this_run = 0
    questions_logged_this_run = 0
    questions_failed_transiently_this_run = 0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Pass all required arguments to the worker
        futures = [executor.submit(process_single_question_worker, q_data, model_id, api_target, api_key)
                   for q_data in questions_to_process]

        for future in concurrent.futures.as_completed(futures):
            try:
                question_data, response_dict = future.result()
                question_id = question_data.get('id', '[Missing ID]')
                questions_processed_this_run += 1

                # Extract metadata and raw response if present
                provider_used = response_dict.get("_provider_used", api_target) if isinstance(response_dict, dict) else api_target
                model_used_for_api = response_dict.get("_model_used_for_api", model_id) if isinstance(response_dict, dict) else model_id
                raw_provider_response = None
                if isinstance(response_dict, dict):
                    raw_provider_response = response_dict.pop("_raw_provider_response", None)
                    # Clean up internal fields from the main response object before logging
                    response_dict.pop("_provider_used", None)
                    response_dict.pop("_model_used_for_api", None)

                # --- Handle response ---
                if response_dict is TRANSIENT_FAILURE_MARKER:
                    questions_failed_transiently_this_run += 1
                    # Logging for transient failure is handled within retry_with_backoff
                    # print(f"  QID {question_id}: Transient failure using {provider_used}, will retry later.") # Removed duplicate log
                else:
                    # 'response_dict' contains the potentially adapted response structure
                    output_entry = {
                        "question_id": question_id,
                        "category": question_data.get('category'),
                        "question": question_data.get('question'),
                        "model": model_name_to_log, # Log canonical name or original API model_id
                        "api_provider": provider_used, # Log the provider used (lowercase)
                        "api_model": model_used_for_api, # Log model name sent to API
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "response": response_dict # Log the standardized (or original if non-Google) response
                    }
                    # Add raw response if it was extracted (primarily for Google)
                    if raw_provider_response is not None:
                         output_entry["raw_response"] = raw_provider_response

                    if 'domain' in question_data:
                        output_entry['domain'] = question_data['domain']

                    # --- Thread-safe writing to output file ---
                    try:
                        with output_lock:
                            with open(output_file, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
                            processed_ids.add(question_id) # Add ID *only* after successful write
                        questions_logged_this_run += 1

                        # --- Simple Progress Indicator ---
                        with PROGRESS_LOCK:
                             PROGRESS_COUNTER += 1
                             if PROGRESS_COUNTER % 10 == 0 or PROGRESS_COUNTER == num_to_process:
                                 elapsed = time.time() - start_time
                                 rate = PROGRESS_COUNTER / elapsed if elapsed > 0 else 0
                                 percent_done = (PROGRESS_COUNTER / num_to_process) * 100
                                 print(f"  Progress: {PROGRESS_COUNTER}/{num_to_process} ({percent_done:.1f}%) logged. Rate: {rate:.2f} Q/s.")

                    except Exception as write_e:
                         print(f"\nCRITICAL ERROR writing entry for Q ID {question_id}: {write_e}")
                         traceback.print_exc()

            except Exception as future_e:
                 print(f"\nCRITICAL ERROR processing future result: {type(future_e).__name__}: {future_e}")
                 traceback.print_exc()

    # --- Final Summary ---
    end_time = time.time()
    duration = end_time - start_time
    print("-" * 30)
    print(f"Finished processing {questions_file}.")
    print(f"  Model requested for API: {model_id}") # Use the API model name here
    if canonical_name:
        print(f"  Canonical name used for logging/filename: {canonical_name}")
    print(f"  API Target used: {api_target.upper()}")
    print(f"  Total questions in file: {total_questions}")
    print(f"  Tasks submitted this run: {num_to_process}")
    print(f"  Tasks completed by workers: {questions_processed_this_run}")
    print(f"  Responses logged this run (Success/PermError): {questions_logged_this_run}")
    print(f"  Tasks failed transiently this run (will retry): {questions_failed_transiently_this_run}")
    permanent_failures_this_run = questions_processed_this_run - questions_logged_this_run - questions_failed_transiently_this_run
    print(f"  Tasks failed permanently this run (or worker error): {permanent_failures_this_run}")
    print(f"  Processing duration: {duration:.2f} seconds")
    final_processed_ids = load_existing_responses(output_file) # Recalculate final count
    print(f"  Total logged entries in file now: {len(final_processed_ids)} / {total_questions}")
    print(f"  Output file: {output_file}")
    print("-" * 30)

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description='Ask questions via API in parallel, supporting multiple providers (OpenAI, OpenRouter, FireworksAI, Chutes, Google), with retries for transient errors. Logs successes and permanent errors.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', help='Model identifier specific to the chosen API provider (e.g., "gpt-4o", "google/gemini-1.5-flash", "accounts/fireworks/models/llama-v3p1-405b-instruct", "THUDM/GLM-4-32B-0414", "gemini-1.0-pro")')
    parser.add_argument('questions_files', nargs='+', help='One or more JSONL question files (must contain "id" and "question" fields).')
    parser.add_argument('--provider', choices=VALID_PROVIDERS, required=True, help='MANDATORY: Specify the API provider to use.')
    parser.add_argument('--canonical-name', help='Optional canonical name for the model used in the output file name and logs (if different from the API model name).')
    parser.add_argument('-w', '--workers', type=int, default=4, help='Number of parallel worker threads for API requests.')
    parser.add_argument('--force-restart', action='store_true', help='Delete existing output file and start fresh (overrides --force-retry-permanent-errors).')
    parser.add_argument('--force-retry-permanent-errors', '--frpe', action='store_true', help='Remove logged permanent errors from the output file before starting, forcing retry.')

    args = parser.parse_args()

    if args.workers < 1:
        print("Error: Number of workers must be at least 1.")
        sys.exit(1)
    if args.force_restart and args.force_retry_permanent_errors:
        print("Warning: --force-restart overrides --force-retry-permanent-errors. Proceeding with full restart.")
        args.force_retry_permanent_errors = False

    # --- Determine API Target and Key based on mandatory --provider ---
    model_id = args.model # This is the API-specific model name
    api_target = args.provider.lower() # Set target directly from argument
    api_key = None
    env_var_used = ""

    print(f"Provider specified: Using '{api_target.upper()}' API.")

    if api_target == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        env_var_used = 'OPENAI_API_KEY'
    elif api_target == "openrouter":
        api_key = os.getenv('OPENROUTER_API_KEY')
        env_var_used = 'OPENROUTER_API_KEY'
    elif api_target == "fireworks":
        api_key = os.getenv('FIREWORKS_API_KEY')
        env_var_used = 'FIREWORKS_API_KEY'
    elif api_target == "chutes":
        api_key = os.getenv('CHUTES_API_TOKEN') # Use CHUTES_API_TOKEN for chutes
        env_var_used = 'CHUTES_API_TOKEN'
    elif api_target == "google":
        api_key = os.getenv('GEMINI_API_KEY') # Use GEMINI_API_KEY for google
        env_var_used = 'GEMINI_API_KEY'
    else:
        # Safeguard, though argparse choices should prevent this
        print(f"Error: Internal error - Invalid provider '{api_target}' selected.")
        sys.exit(1)


    # --- Validate API Key ---
    if not api_key:
        print(f"Error: Required API key environment variable '{env_var_used}' is not set for the chosen API provider '{api_target.upper()}'.")
        sys.exit(1)

    # --- Log Final Config ---
    print(f"Using API Target: {api_target.upper()}")
    print(f"Using API Key from env var: {env_var_used}")
    if api_target == "openrouter":
        print(f"OpenRouter Referrer: {os.getenv('OPENROUTER_REFERRER', 'http://localhost')}")
        print(f"OpenRouter Title: {os.getenv('OPENROUTER_TITLE', 'AI Question Answering Script')}")
    print(f"Using {args.workers} parallel worker(s).")
    # Log API model name and canonical name if provided
    print(f"Using API model name: {model_id}")
    if args.canonical_name:
        print(f"Using canonical name for logging/filename: {args.canonical_name}")
    else:
        print(f"Using API model name for logging/filename (no canonical name provided).")


    # --- Process Files ---
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
            # Pass the determined api_target, api_key, the API model_id, and optional canonical_name
            process_questions(
                questions_file=questions_file,
                model_id=model_id, # Pass the API-specific model name
                canonical_name=args.canonical_name, # Pass canonical name (can be None)
                api_target=api_target,
                api_key=api_key,
                num_workers=args.workers,
                force_restart=args.force_restart,
                force_retry_permanent=args.force_retry_permanent_errors
                )
            files_processed_count += 1
        except KeyboardInterrupt:
             print("\nInterrupted by user. Stopping.")
             files_failed_count += 1 # Count current file as failed/incomplete
             break # Exit loop over files
        except Exception as e:
            files_failed_count += 1
            print(f"\n!!! CRITICAL UNEXPECTED ERROR processing {questions_file} !!!")
            print(f"Error: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            print("Attempting to continue with next file...")
        file_end_time = time.time()
        print(f"=== Finished processing file: {questions_file} (Duration: {file_end_time - file_start_time:.2f} seconds) ===")

    # --- Overall Summary ---
    total_end_time = time.time()
    print("\n" + "="*40)
    print("Overall Summary:")
    print(f"  API Model used: {model_id}") # Display the model name passed to the API
    if args.canonical_name:
        print(f"  Canonical Name logged: {args.canonical_name}")
    print(f"  API Provider used: {api_target.upper()}")
    print(f"  Total files attempted: {len(args.questions_files)}")
    print(f"  Files processed successfully: {files_processed_count}")
    print(f"  Files failed/skipped/interrupted: {files_failed_count}")
    print(f"  Total execution time: {total_end_time - total_start_time:.2f} seconds")
    print("="*40)

    if files_failed_count > 0:
        sys.exit(1) # Exit with error status if any files failed

if __name__ == "__main__":
    main()
