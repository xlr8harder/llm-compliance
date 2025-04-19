# ask.py
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
def load_questions(file_path):
    """Load questions from a JSONL file."""
    questions = []
    path = Path(file_path)
    if not path.is_file():
        print(f"Error: Input questions file not found: {file_path}", file=sys.stderr)
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
                            print(f"Warning: Skipping line {i+1} in questions file {file_path}: missing 'id' or 'question'.", file=sys.stderr)
                            skipped_count += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {i+1} in questions file {file_path}: {e}. Skipping line.", file=sys.stderr)
                    skipped_count += 1
        print(f"Loaded {valid_count} valid questions from {file_path}.")
        if skipped_count > 0: print(f"Skipped {skipped_count} invalid/malformed lines.")
        return questions, True
    except Exception as e:
        print(f"Error reading questions file {file_path}: {e}", file=sys.stderr)
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
                    print(f"Warning: Line {lines_read} missing 'question_id'. Skipping.", file=sys.stderr)
                    parse_warnings += 1
            except (json.JSONDecodeError) as e:
                print(f"Warning: Could not parse line {lines_read}: {str(e)}. Line: {stripped_line}", file=sys.stderr)
                parse_warnings += 1
            except Exception as e:
                print(f"Warning: Error processing line {lines_read}: {type(e).__name__}: {str(e)}. Line: {stripped_line}", file=sys.stderr)
                parse_warnings += 1
    print(f"Finished loading. Lines processed: {lines_read}")
    print(f"  - Found {len(processed_ids)} existing logged entries (Successes or Permanent Errors).")
    if parse_warnings > 0:
        print(f"  - Warnings during loading: {parse_warnings}")
    return processed_ids

def cleanup_permanent_errors(output_file):
    """
    Rewrites the output file, keeping ONLY successful responses.
    Also removes responses where finish_reason is 'error'.
    """
    input_path = Path(output_file) # Ensure it's a Path object
    if not input_path.exists():
        print("Cleanup requested, but output file does not exist.")
        return False
    temp_file_path = None
    lines_read, lines_written, api_errors_skipped, empty_skipped, finish_reason_error_skipped, corrupted_skipped = 0, 0, 0, 0, 0, 0
    print(f"\n--- Cleaning up permanent errors in: {input_path} ---")
    print("--- Keeping ONLY successful responses (excluding finish_reason='error'). ---")
    success = True # Track if process completes without errors
    try:
        # Use NamedTemporaryFile correctly within a 'with' statement
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=input_path.parent, suffix='.tmp') as temp_f:
            temp_file_path = Path(temp_f.name) # Get the path
            with open(input_path, 'r', encoding='utf-8') as original_f:
                for i, line in enumerate(original_f):
                    lines_read = i + 1
                    stripped_line = line.strip()

                    if not stripped_line:
                        continue # Skip empty lines

                    keep_line = False
                    try:
                        entry = json.loads(stripped_line)
                        # Check for required fields before processing
                        response_data = entry.get("response")
                        question_id = entry.get("question_id")

                        if not question_id:
                            corrupted_skipped += 1
                            print(f"Warning: Line {lines_read} missing 'question_id' during cleanup, skipping.", file=sys.stderr)
                            continue # Skip this line

                        if response_data is None: # Handle case where 'response' key is missing
                            corrupted_skipped +=1
                            print(f"Warning: Line {lines_read} (QID: {question_id}) missing 'response' data during cleanup, skipping.", file=sys.stderr)
                            continue # Skip this line

                        # Now check the content of the response using the helper functions
                        # which expect the standardized format
                        is_api_error = is_permanent_api_error(response_data)
                        is_empty = is_empty_content_response(response_data)

                        # Check for finish_reason == 'error'
                        is_finish_reason_error = False
                        try:
                            if isinstance(response_data, dict) and response_data.get('choices') and isinstance(response_data['choices'], list) and len(response_data['choices']) > 0:
                                if isinstance(response_data['choices'][0], dict) and response_data['choices'][0].get('finish_reason') == 'error':
                                    is_finish_reason_error = True
                        except (KeyError, IndexError, TypeError):
                            pass # Ignore structure errors here, focus on finish_reason

                        # Determine if line should be skipped
                        if is_api_error:
                            api_errors_skipped += 1
                        elif is_empty:
                            empty_skipped += 1
                        elif is_finish_reason_error:
                            finish_reason_error_skipped += 1
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
                                 # print(f"Warning: Line {lines_read} (QID: {question_id}) lacks valid content structure, skipping.", file=sys.stderr)

                        if keep_line:
                            temp_f.write(line) # Write the original line
                            lines_written += 1

                    except (json.JSONDecodeError):
                        corrupted_skipped += 1
                        print(f"Warning: Corrupted line {lines_read} encountered during cleanup (JSONDecodeError), skipping.", file=sys.stderr)
                        success = False # Mark as failed if corruption occurs
                    except Exception as e_inner: # Catch other potential errors processing a line
                         corrupted_skipped += 1
                         print(f"Warning: Error processing line {lines_read} during cleanup: {type(e_inner).__name__}. Skipping.", file=sys.stderr)
                         success = False # Mark as failed


        # --- Post-processing and replacement ---
        if lines_read > 0:
             print(f"\nCleanup processed {lines_read} lines.")
             print(f"  - Kept {lines_written} successful responses.")
             print(f"  - Skipped {api_errors_skipped} permanent API errors.")
             print(f"  - Skipped {empty_skipped} empty content responses.")
             print(f"  - Skipped {finish_reason_error_skipped} with finish_reason='error'.")
             print(f"  - Skipped {corrupted_skipped} corrupted/unparseable/invalid structure lines.")

             # Replace only if the process was successful and changes were made
             if success and (lines_written != lines_read or corrupted_skipped > 0):
                 # Double check if actual changes occurred that require writing
                 if lines_written > 0 or api_errors_skipped > 0 or empty_skipped > 0 or finish_reason_error_skipped > 0 or corrupted_skipped > 0 :
                     print(f"Attempting to atomically replace original file...")
                     try:
                         # Ensure temp file is closed before replacing (it is by end of 'with')
                         os.replace(temp_file_path, input_path)
                         print(f"Successfully replaced {input_path}.")
                         temp_file_path = None # Avoid deletion in finally block
                         print("--- Cleanup Finished Successfully ---")
                         return True
                     except OSError as e_replace:
                          print(f"!!! ERROR replacing file {input_path} with {temp_file_path}: {e_replace} !!!", file=sys.stderr)
                          return False # Indicate failure
                 else:
                     print("No valid lines kept and no errors/empty found to remove. Original file likely empty or contained only corrupted lines.")
                     # No need to replace if nothing changed
                     print("--- Cleanup Finished (No Changes Made) ---")
                     return True
             elif not success:
                 print("Cleanup process encountered errors. Original file not replaced.", file=sys.stderr)
                 return False
             else: # Success and no changes needed
                 print("No errors found to remove. Original file unchanged.")
                 print("--- Cleanup Finished (No Changes Made) ---")
                 return True

        else: # Original file was empty or only had blank lines
             print("Original file was empty or contained no processable lines. No cleanup performed.")
             return True # Not an error state

    except Exception as e:
        print(f"!!! ERROR during cleanup process: {type(e).__name__}: {e} !!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False
    finally:
        # Clean up temp file if replacement failed or wasn't attempted
        if temp_file_path and temp_file_path.exists():
            print(f"Cleaning up temporary file: {temp_file_path}")
            try:
                temp_file_path.unlink()
            except OSError as unlink_e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {unlink_e}", file=sys.stderr)


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
                    # Return the detailed error if available
                    return {"error": error_info if error_info else result.get("error_detail", f"Permanent Client Error {status_code}")}


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
        # Model version might be at top level or within usageMetadata historically
        if 'modelVersion' in google_response:
             adapted['model'] = google_response['modelVersion']
        elif 'usageMetadata' in google_response and 'modelVersion' in google_response['usageMetadata']:
             adapted['model'] = google_response['usageMetadata']['modelVersion']

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
                # Combine text from all parts, if multiple exist (though usually just one)
                choice['message']['content'] = "".join([part.get('text', '') for part in candidate['content']['parts'] if 'text' in part])


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
             # If no candidates and no prompt feedback block, it's likely an error (e.g., API issue)
             # Check for top-level error message from Google if available
             top_level_error = google_response.get('error', {})
             error_message = top_level_error.get('message', 'Invalid response structure: Missing candidates or promptFeedback')
             error_code = top_level_error.get('code')
             adapted['error'] = {'message': error_message, 'type': 'invalid_response', 'code': error_code}


    except (KeyError, IndexError, TypeError, AttributeError) as e:
        print(f"Error adapting Google response: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
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
        # Google safety settings (optional, can be adjusted)
        # Setting to BLOCK_NONE attempts to disable the safety filter
        data["safetySettings"] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            # {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"} # If needed
        ]

    else:
        print(f"ERROR: Unknown api_target '{api_target}' in ask_question.", file=sys.stderr)
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
    # Pass question_id for better logging during retries
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

# --- Parameter Detection and Annotation ---

def guess_and_test_parameters(canonical_model_name):
    """Guesses provider and API model, tests OpenRouter first, then prefix provider."""
    print(f"Attempting to guess provider and API model for '{canonical_model_name}'...")

    if not isinstance(canonical_model_name, str):
        raise ValueError("Invalid canonical model name for guessing.")

    # --- Guess API Model Name (for direct provider calls) ---
    api_model_guess = canonical_model_name
    if '/' in canonical_model_name:
        api_model_guess = canonical_model_name.split('/')[-1]

    # --- Determine Prefix Provider (if any) ---
    prefix_provider = None
    prefix_map = {"openai": "openai", "google": "google"} # Add more if needed
    if '/' in canonical_model_name:
        prefix = canonical_model_name.split('/')[0].lower()
        prefix_provider = prefix_map.get(prefix)

    # --- Attempt 1: Test OpenRouter ---
    provider_to_test = "openrouter"
    model_for_or_call = canonical_model_name # OpenRouter usually uses the full name
    print(f"  Attempt 1: Testing {provider_to_test.upper()} with model '{model_for_or_call}'...")
    or_key, or_env_var = get_api_key_for_provider(provider_to_test, fail_if_missing=False)

    if or_key:
        try:
            test_response = ask_question("Output the single word 'test' and nothing else.", "PARAM_TEST_OR", model_for_or_call, provider_to_test, or_key)
            if test_response is not TRANSIENT_FAILURE_MARKER and 'error' not in test_response and test_response.get('choices'):
                print(f"    Success! Using Provider='{provider_to_test}', API Model='{model_for_or_call}'")
                return provider_to_test, model_for_or_call
            else:
                print(f"    OpenRouter test failed. Response/Reason: {str(test_response)[:200]}")
        except Exception as e:
            print(f"    OpenRouter test failed with exception: {e}")
    else:
        print(f"    Skipping OpenRouter test: Environment variable '{or_env_var}' not set.")

    # --- Attempt 2: Test Prefix Provider (if applicable) ---
    if prefix_provider and prefix_provider != "openrouter":
        provider_to_test = prefix_provider
        model_for_prefix_call = api_model_guess # Use the shortened name
        print(f"  Attempt 2: Testing {provider_to_test.upper()} with model '{model_for_prefix_call}'...")
        prefix_key, prefix_env_var = get_api_key_for_provider(provider_to_test, fail_if_missing=True) # Fail if key missing here

        try:
            test_response = ask_question("Output the single word 'test' and nothing else.", "PARAM_TEST_PREFIX", model_for_prefix_call, provider_to_test, prefix_key)
            if test_response is not TRANSIENT_FAILURE_MARKER and 'error' not in test_response and test_response.get('choices'):
                 print(f"    Success! Using Provider='{provider_to_test}', API Model='{model_for_prefix_call}'")
                 return provider_to_test, model_for_prefix_call
            else:
                 print(f"    Prefix provider ({provider_to_test}) test failed. Response/Reason: {str(test_response)[:200]}")
                 # Proceed to failure if prefix provider test fails
        except Exception as e:
            print(f"    Prefix provider ({provider_to_test}) test failed with exception: {e}")
            # Proceed to failure
    else:
        print("  Skipping Attempt 2: No suitable prefix provider found or it was OpenRouter.")


    # --- Failure ---
    raise RuntimeError(f"Could not confirm working provider/model parameters for '{canonical_model_name}' after testing.")


def detect_and_verify_parameters(output_file_path):
    """Reads the first entry, detects params, guesses/tests if needed."""
    print(f"Detecting parameters from first valid entry in: {output_file_path}")
    path = Path(output_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Output file for detection not found: {output_file_path}")

    detected_params = {
        'category': None,
        'api_provider': None,
        'api_model': None,
        'canonical_name': None,
        'questions_file_path': None,
        'metadata_was_missing': False,
        'output_file_path': str(output_file_path) # Store for logging
    }

    try:
        with path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    entry = json.loads(line.strip())
                    # Extract required category
                    detected_params['category'] = entry.get('category')
                    if not detected_params['category']:
                        raise ValueError(f"First valid entry (line {i+1}) in {output_file_path} is missing mandatory 'category' field.")

                    # Extract optional fields
                    detected_params['api_provider'] = entry.get('api_provider')
                    detected_params['api_model'] = entry.get('api_model')
                    detected_params['canonical_name'] = entry.get('model') # 'model' holds canonical name

                    # Check if metadata was missing
                    if not detected_params['api_provider'] or not detected_params['api_model']:
                        print("  Metadata (api_provider/api_model) missing. Attempting guess and test...")
                        detected_params['metadata_was_missing'] = True
                        # Use canonical name for guessing
                        if not detected_params['canonical_name']:
                             raise ValueError(f"Cannot guess parameters: 'model' (canonical name) field is missing in entry (line {i+1}) where metadata is needed.")

                        # Call guess and test - raises RuntimeError on failure
                        confirmed_provider, confirmed_api_model = guess_and_test_parameters(detected_params['canonical_name'])
                        detected_params['api_provider'] = confirmed_provider
                        detected_params['api_model'] = confirmed_api_model
                    else:
                         print(f"  Found existing metadata: Provider='{detected_params['api_provider']}', API Model='{detected_params['api_model']}'")


                    # Construct questions file path
                    q_dir = Path("questions")
                    detected_params['questions_file_path'] = q_dir / f"{detected_params['category']}.jsonl"
                    print(f"  Derived Questions File: {detected_params['questions_file_path']}")

                    # Use the first valid entry found
                    return detected_params

                except json.JSONDecodeError:
                    continue # Skip corrupted lines silently when just detecting params
                except ValueError as ve: # Catch specific errors from our logic
                    raise ve

        # If loop finishes without finding a valid entry
        raise ValueError(f"Could not find any valid JSON entries in {output_file_path} to detect parameters.")

    except Exception as e:
        # Catch file read errors or other exceptions
        raise RuntimeError(f"Failed to detect parameters from {output_file_path}: {e}")


def annotate_output_file(output_file_path, confirmed_provider, confirmed_api_model):
    """Rewrites the output file, adding/updating api_provider and api_model."""
    input_path = Path(output_file_path)
    if not input_path.is_file():
        print(f"Error: Cannot annotate, file not found: {output_file_path}", file=sys.stderr)
        return False

    temp_file_path = None
    lines_read = 0
    lines_written = 0
    error_count = 0
    success = True

    print(f"\n--- Annotating file with confirmed parameters: {input_path} ---")
    print(f"    Provider='{confirmed_provider}', API Model='{confirmed_api_model}'")

    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False,
                                         dir=input_path.parent, suffix='.annot.tmp') as temp_f:
            temp_file_path = Path(temp_f.name)
            with input_path.open('r', encoding='utf-8') as infile:
                for i, line in enumerate(infile):
                    lines_read = i + 1
                    try:
                        entry = json.loads(line)
                        # Add/overwrite the fields
                        entry['api_provider'] = confirmed_provider
                        entry['api_model'] = confirmed_api_model
                        temp_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        lines_written += 1
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping corrupted JSON line {lines_read} during annotation.", file=sys.stderr)
                        error_count += 1
                        continue
                    except Exception as e:
                        print(f"Error processing line {lines_read} during annotation: {e}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
                        error_count += 1
                        success = False # Mark annotation as failed

        # --- Replacement ---
        if not success or error_count > 0:
            print(f"Annotation process encountered errors ({error_count}). Original file not replaced.", file=sys.stderr)
            success = False
        else:
            print(f"Annotation complete ({lines_written}/{lines_read} lines written). Replacing original file...")
            try:
                os.replace(temp_file_path, input_path)
                print(f"Successfully replaced {input_path} with annotated version.")
                temp_file_path = None # Prevent deletion
            except OSError as e_replace:
                print(f"!!! ERROR replacing file {input_path} with {temp_file_path}: {e_replace} !!!", file=sys.stderr)
                success = False

    except Exception as e:
        print(f"!!! CRITICAL ERROR during annotation process for {output_file_path}: {e} !!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        success = False
    finally:
        if temp_file_path and temp_file_path.exists():
            print(f"Cleaning up temporary annotation file: {temp_file_path}")
            try:
                temp_file_path.unlink()
            except OSError as unlink_e:
                print(f"Warning: Could not delete temporary annotation file {temp_file_path}: {unlink_e}", file=sys.stderr)

    return success


# --- Main Processing Logic ---
def process_questions(questions_file, model_id, canonical_name, api_target, api_key, num_workers, force_restart=False, force_retry_permanent=False, output_file_path_override=None):
    """
    Process questions in parallel, save responses, handle resumes and cleanup.
    Takes an optional output_file_path_override for --detect mode.
    """
    global PROGRESS_COUNTER
    PROGRESS_COUNTER = 0 # Reset counter for each file

    output_dir = Path("responses")
    output_dir.mkdir(exist_ok=True)

    if output_file_path_override:
         # Use the provided path directly (from --detect mode)
         output_file = Path(output_file_path_override)
         # Attempt to reconstruct original questions filename base and model suffix for context
         try:
             base_name = output_file.stem
             # Find the last underscore separating category/qfile from model info
             split_index = base_name.rfind('_')
             if split_index != -1:
                 questions_filename = base_name[:split_index]
                 safe_output_model_name = base_name[split_index+1:]
             else: # Fallback if no underscore found
                 questions_filename = "unknown_questions"
                 safe_output_model_name = base_name
         except Exception: # Catch any errors during reconstruction
             questions_filename = "unknown_questions"
             safe_output_model_name = "unknown_model"
         print(f"Using specified output file: {output_file}")
    else:
         # Construct output path as before for normal mode
         questions_filename = Path(questions_file).stem
         output_model_name = canonical_name if canonical_name else model_id
         safe_output_model_name = output_model_name.replace('/', '_').replace(':', '-')
         output_file = output_dir / f"{questions_filename}_{safe_output_model_name}.jsonl"
         print(f"Using constructed output file: {output_file}")


    questions, loaded_ok = load_questions(questions_file)
    if not loaded_ok:
        # Error logged in load_questions
        raise IOError(f"Failed to load questions from {questions_file}") # Raise error to stop processing this file
    total_questions = len(questions)

    # --- Resume / Restart Logic ---
    # Note: In --detect mode, cleanup_permanent_errors was already run *before* this function.
    # Here, we just need to load IDs of successfully processed items *after* cleanup.
    processed_ids = set()
    if output_file.exists():
         if not force_restart: # force_restart would have deleted the file in --detect mode prep
              # Load existing IDs *after* potential cleanup (FRPE)
              processed_ids = load_existing_responses(output_file)
              if force_retry_permanent:
                  print(f"Resuming FRPE run. Found {len(processed_ids)} entries remaining after cleanup in '{output_file}'.")
              else: # Should not happen if FRPE is mandatory for --detect
                  print(f"Resuming normal run. Found {len(processed_ids)} completed entries in '{output_file}'.")
         # If force_restart was true, processed_ids remains empty.
    else:
         # File doesn't exist (either fresh run or deleted by force_restart)
         print(f"No existing output file found at '{output_file}'. Starting fresh or continuing FRPE.")


    # --- Filter Questions ---
    questions_to_process = []
    for q_data in questions:
        q_id = q_data.get('id')
        if not q_id:
            print(f"Warning: Question missing 'id', skipping: {q_data.get('question', '')[:50]}...", file=sys.stderr)
            continue
        # Only process if ID is NOT in the set of already successfully processed items
        if q_id not in processed_ids:
            questions_to_process.append(q_data)

    num_to_process = len(questions_to_process)
    num_skipped = total_questions - num_to_process
    print(f"\nTotal questions in source file: {total_questions}")
    print(f"Already processed and kept: {num_skipped}")
    print(f"Questions to process this run: {num_to_process}")
    if num_to_process == 0:
        print("No questions remaining to process for this configuration.")
        return # Nothing to do for this file

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
                            # Append to the potentially cleaned/annotated output file
                            with open(output_file, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
                            # Add to processed_ids immediately after write? Or load at end?
                            # Loading at end is safer if write fails partially.
                            # processed_ids.add(question_id) # Maybe remove this
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
                         print(f"\nCRITICAL ERROR writing entry for Q ID {question_id}: {write_e}", file=sys.stderr)
                         traceback.print_exc(file=sys.stderr)
                         # Consider how to handle write errors - maybe stop the run?
                         # For now, just log and continue, but this entry might be lost/partial.

            except Exception as future_e:
                 print(f"\nCRITICAL ERROR processing future result: {type(future_e).__name__}: {future_e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)
                 # This indicates a problem in the worker or result handling

    # --- Final Summary ---
    end_time = time.time()
    duration = end_time - start_time
    print("-" * 30)
    print(f"Finished processing questions from {questions_file}.") # Log source questions file
    print(f"  Output file: {output_file}") # Log target output file
    print(f"  Model requested for API: {model_id}") # Use the API model name here
    if canonical_name:
        print(f"  Canonical name used for logging/filename: {canonical_name}")
    print(f"  API Target used: {api_target.upper()}")
    print(f"  Total questions in source file: {total_questions}")
    print(f"  Tasks submitted this run: {num_to_process}")
    print(f"  Tasks completed by workers: {questions_processed_this_run}")
    print(f"  Responses logged this run (Success/PermError): {questions_logged_this_run}")
    print(f"  Tasks failed transiently this run (will retry later if run again): {questions_failed_transiently_this_run}")
    permanent_failures_this_run = questions_processed_this_run - questions_logged_this_run - questions_failed_transiently_this_run
    print(f"  Tasks failed permanently this run (or worker error): {permanent_failures_this_run}")
    print(f"  Processing duration: {duration:.2f} seconds")
    # Recalculate final count from file for accuracy after run completes
    final_processed_ids = load_existing_responses(output_file)
    print(f"  Total logged entries in file now: {len(final_processed_ids)} / {total_questions}") # Compare to total source questions
    print("-" * 30)


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description='Ask questions via API in parallel or reprocess existing outputs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Mode Selection ---
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--normal', action='store_true', help='Normal mode: Run questions from input files.')
    mode_group.add_argument('--detect', metavar='OUTPUT_FILE', help='Detect mode: Reprocess errors in an existing output file, detecting parameters.')

    # --- Normal Mode Arguments ---
    parser.add_argument('--model', help='[Normal Mode / Override in Detect Mode] Model identifier specific to the chosen API provider.')
    parser.add_argument('--questions_files', nargs='+', help='[Normal Mode] One or more JSONL question files.')
    parser.add_argument('--provider', choices=VALID_PROVIDERS, help='[Normal Mode Required / Override in Detect Mode] Specify the API provider.')

    # --- Shared Arguments ---
    parser.add_argument('--canonical-name', help='[Optional in Both Modes] Canonical name for model in output/logs (overrides detected/inferred).')
    parser.add_argument('-w', '--workers', type=int, default=4, help='Number of parallel worker threads.')
    parser.add_argument('--force-restart', action='store_true', help='Delete existing output file and start fresh (overrides --frpe).')
    parser.add_argument('--frpe', '--force-retry-permanent-errors', action='store_true', help='Remove logged errors from output file before starting, forcing retry.')

    args = parser.parse_args()

    # --- Validate Mode-Specific Arguments ---
    if args.normal:
        if not args.model or not args.questions_files or not args.provider:
            parser.error("--normal mode requires --model, --questions_files, and --provider.")
        if args.detect: # Should be caught by mutually_exclusive_group, but double-check
             parser.error("Cannot use --detect with --normal.")
        # Warn if --frpe is used without --detect
        if args.frpe and not args.detect:
             print("Warning: --frpe is typically used with --detect mode. Using it in --normal mode will clean the target output file before writing new results.")

    elif args.detect:
        if not args.frpe:
            parser.error("--detect mode requires --frpe.")
        # Allow --model override, --provider override is handled below
        if args.questions_files:
             parser.error("Cannot specify --questions_files in --detect mode (derived from category).")

    if args.workers < 1:
        parser.error("Number of workers must be at least 1.")

    # FRPE/Restart interaction: If --force-restart, FRPE's cleanup is skipped because the file is deleted anyway.
    if args.force_restart and args.frpe:
        print("Note: --force-restart is active; output file will be deleted before processing, skipping specific FRPE cleanup step.")
        # No need to disable args.frpe, the logic handles it


    # --- Execute Selected Mode ---
    overall_success = True
    try:
        if args.normal:
            # --- Normal Mode Execution ---
            print("Running in NORMAL mode.")
            api_target = args.provider.lower()
            model_id = args.model # This is the API model name in normal mode
            canonical_name = args.canonical_name if args.canonical_name else None

            api_key, env_var_used = get_api_key_for_provider(api_target) # Raises ValueError if key missing
            log_final_config(api_target, env_var_used, args.workers, model_id, canonical_name)

            total_start_time = time.time()
            files_processed_count, files_failed_count = 0, 0

            for questions_file in args.questions_files:
                output_file_path = None # Determine path inside process_questions
                if args.force_restart:
                     # Construct potential output path to delete it
                     output_dir = Path("responses")
                     questions_filename = Path(questions_file).stem
                     output_model_name_part = canonical_name if canonical_name else model_id
                     safe_output_model_name = output_model_name_part.replace('/', '_').replace(':', '-')
                     output_file_to_delete = output_dir / f"{questions_filename}_{safe_output_model_name}.jsonl"
                     if output_file_to_delete.exists():
                         print(f"Force restart: Deleting existing output file '{output_file_to_delete}'")
                         try:
                             output_file_to_delete.unlink()
                         except OSError as e:
                             print(f"Error deleting file {output_file_to_delete}: {e}. Skipping processing for {questions_file}.", file=sys.stderr)
                             files_failed_count += 1
                             overall_success = False
                             continue # Skip to next questions file

                print(f"\n=== Starting processing for file: {questions_file} ===")
                file_start_time = time.time()
                if not Path(questions_file).is_file():
                    print(f"Error: Input file not found: {questions_file}. Skipping.", file=sys.stderr)
                    files_failed_count += 1
                    overall_success = False
                    continue
                try:
                    # Run cleanup *before* processing if --frpe is set and --force-restart is not
                    if args.frpe and not args.force_restart:
                         # Construct potential output path to clean it
                         output_dir = Path("responses")
                         questions_filename = Path(questions_file).stem
                         output_model_name_part = canonical_name if canonical_name else model_id
                         safe_output_model_name = output_model_name_part.replace('/', '_').replace(':', '-')
                         output_file_to_clean = output_dir / f"{questions_filename}_{safe_output_model_name}.jsonl"
                         if not cleanup_permanent_errors(output_file_to_clean):
                             print(f"Error: FRPE cleanup failed for {output_file_to_clean}. Skipping processing for {questions_file}.", file=sys.stderr)
                             files_failed_count += 1
                             overall_success = False
                             continue # Skip to next questions file


                    process_questions(
                        questions_file=questions_file,
                        model_id=model_id, # API model name
                        canonical_name=canonical_name,
                        api_target=api_target,
                        api_key=api_key,
                        num_workers=args.workers,
                        force_restart=args.force_restart, # Pass flag
                        force_retry_permanent=args.frpe, # Pass flag
                        output_file_path_override=None # Not used in normal mode
                    )
                    files_processed_count += 1
                except KeyboardInterrupt:
                     print("\nInterrupted by user. Stopping.")
                     files_failed_count += (len(args.questions_files) - files_processed_count)
                     overall_success = False
                     break
                except Exception as e:
                    files_failed_count += 1
                    print(f"\n!!! CRITICAL UNEXPECTED ERROR processing {questions_file} !!!", file=sys.stderr)
                    print(f"Error: {type(e).__name__}: {str(e)}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    print("Attempting to continue with next file...")
                    overall_success = False
                file_end_time = time.time()
                print(f"=== Finished processing file: {questions_file} (Duration: {file_end_time - file_start_time:.2f} seconds) ===")

            # --- Overall Summary for Normal Mode ---
            total_end_time = time.time()
            print("\n" + "="*40)
            print("Overall Summary (Normal Mode):")
            print(f"  API Model used: {model_id}")
            if canonical_name: print(f"  Canonical Name logged: {canonical_name}")
            print(f"  API Provider used: {api_target.upper()}")
            print(f"  Total files attempted: {len(args.questions_files)}")
            print(f"  Files processed successfully: {files_processed_count}")
            print(f"  Files failed/skipped/interrupted: {files_failed_count}")
            print(f"  Total execution time: {total_end_time - total_start_time:.2f} seconds")
            print("="*40)


        elif args.detect:
            # --- Detect Mode Execution ---
            print("Running in DETECT mode.")
            output_file_to_detect = args.detect
            output_file_path = Path(output_file_to_detect)

            if args.force_restart and output_file_path.exists():
                print(f"Force restart: Deleting existing output file '{output_file_path}' before detection.")
                try:
                    output_file_path.unlink()
                    print("Existing file removed.")
                except OSError as e:
                    print(f"FATAL ERROR: Could not remove file {output_file_path} for restart: {e}", file=sys.stderr)
                    sys.exit(1)
            elif not output_file_path.exists():
                 print(f"Error: Output file specified for --detect does not exist: {output_file_path}", file=sys.stderr)
                 sys.exit(1)

            # 1. Detect/Verify Parameters
            detected_params = detect_and_verify_parameters(output_file_to_detect) # Raises exceptions on failure

            # 2. Annotate if necessary
            if detected_params['metadata_was_missing']:
                if not annotate_output_file(output_file_to_detect, detected_params['api_provider'], detected_params['api_model']):
                    print(f"FATAL ERROR: Failed to annotate output file {output_file_to_detect}.", file=sys.stderr)
                    sys.exit(1) # Stop if annotation fails

            # 3. Apply Overrides
            # If --provider is given, it overrides detected/guessed provider
            final_api_target = args.provider.lower() if args.provider else detected_params['api_provider']
            # If --model is given, it overrides detected/guessed API model name
            final_api_model = args.model if args.model else detected_params['api_model']
            # If --canonical-name is given, it overrides detected canonical name
            final_canonical_name = args.canonical_name if args.canonical_name else detected_params['canonical_name']
            questions_file_path = detected_params['questions_file_path'] # Cannot be overridden

            # 4. Get final API Key
            final_api_key, final_env_var_used = get_api_key_for_provider(final_api_target) # Raises ValueError if key missing

            # 5. Log Final Config
            log_final_config(final_api_target, final_env_var_used, args.workers, final_api_model, final_canonical_name, detected_params)

            # 6. Clean Errors (FRPE) - output_file_to_detect path is used here
            # The --frpe flag guarantees this runs; cleanup_permanent_errors handles logging
            # Skip cleanup if force_restart was used (file already deleted)
            if not args.force_restart:
                print(f"\nRunning FRPE cleanup on: {output_file_to_detect}")
                if not cleanup_permanent_errors(output_file_to_detect):
                    print(f"FATAL ERROR: Cleanup of permanent errors failed for {output_file_to_detect}.", file=sys.stderr)
                    sys.exit(1)
            else:
                print("\nSkipping FRPE cleanup because --force-restart was used.")

            # 7. Process Questions (using detected/overridden params)
            print(f"\n=== Starting reprocessing for detected file: {output_file_to_detect} ===")
            process_questions(
                questions_file=questions_file_path,
                model_id=final_api_model,
                canonical_name=final_canonical_name,
                api_target=final_api_target,
                api_key=final_api_key,
                num_workers=args.workers,
                force_restart=False, # Restart already handled, cleanup done
                force_retry_permanent=True, # Implicitly True due to --frpe requirement
                output_file_path_override=output_file_to_detect # Ensure output goes to the correct file
            )
            # Summary is printed within process_questions

    except (ValueError, RuntimeError, IOError, FileNotFoundError) as e:
        # Catch errors raised during mode execution setup/validation
        print(f"\n!!! SETUP/CONFIGURATION ERROR !!!", file=sys.stderr)
        print(f"Error: {type(e).__name__}: {str(e)}", file=sys.stderr)
        # traceback.print_exc(file=sys.stderr) # Optional: full traceback for setup errors
        overall_success = False
    except KeyboardInterrupt:
        print("\nInterrupted by user during setup or file loop.")
        overall_success = False
    except Exception as e:
        # Catch any other unexpected critical errors in main
        print(f"\n!!! CRITICAL UNEXPECTED ERROR in main execution !!!", file=sys.stderr)
        print(f"Error: {type(e).__name__}: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        overall_success = False

    sys.exit(0 if overall_success else 1)


def get_api_key_for_provider(provider_name, fail_if_missing=True):
    """Gets the API key and env var name for a given provider."""
    api_key = None
    env_var = ""
    provider_name = provider_name.lower() # Ensure lowercase comparison

    if provider_name == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        env_var = 'OPENAI_API_KEY'
    elif provider_name == "openrouter":
        api_key = os.getenv('OPENROUTER_API_KEY')
        env_var = 'OPENROUTER_API_KEY'
    elif provider_name == "fireworks":
        api_key = os.getenv('FIREWORKS_API_KEY')
        env_var = 'FIREWORKS_API_KEY'
    elif provider_name == "chutes":
        api_key = os.getenv('CHUTES_API_TOKEN')
        env_var = 'CHUTES_API_TOKEN'
    elif provider_name == "google":
        api_key = os.getenv('GEMINI_API_KEY')
        env_var = 'GEMINI_API_KEY'
    else:
        raise ValueError(f"Invalid provider name '{provider_name}' for API key lookup.")

    if not api_key and fail_if_missing:
        raise ValueError(f"Required API key environment variable '{env_var}' is not set for provider '{provider_name}'.")

    return api_key, env_var

def log_final_config(api_target, env_var_used, workers, model_id, canonical_name, detected_params=None):
    """Logs the final configuration parameters being used."""
    print(f"\n--- Final Configuration ---")
    if detected_params:
         print(f"Mode: Detect (Source Output: {detected_params.get('output_file_path','N/A')})")
         print(f"  Detected Category: {detected_params.get('category','N/A')} -> Questions File: {detected_params.get('questions_file_path','N/A')}")
         print(f"  Detected Provider: {detected_params.get('api_provider','N/A')}")
         print(f"  Detected API Model: {detected_params.get('api_model','N/A')}")
         print(f"  Detected Canon. Name: {detected_params.get('canonical_name','N/A')}")
         if detected_params.get('metadata_was_missing'):
             print("  (Provider/API Model were guessed and tested)")
         print("--- Overrides Applied (if any) ---")

    print(f"API Target: {api_target.upper()}")
    print(f"API Key Env Var: {env_var_used}")
    if api_target == "openrouter":
        print(f"OpenRouter Referrer: {os.getenv('OPENROUTER_REFERRER', 'http://localhost')}")
        print(f"OpenRouter Title: {os.getenv('OPENROUTER_TITLE', 'AI Question Answering Script')}")
    print(f"API Model Name: {model_id}")
    if canonical_name:
        print(f"Canonical Name (Logging/Filename): {canonical_name}")
    else:
        print(f"Canonical Name: (Not specified, using API model name)")
    print(f"Workers: {workers}")
    print(f"-------------------------\n")


if __name__ == "__main__":
    main()
