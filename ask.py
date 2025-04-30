# ask.py
import argparse
import datetime
import json
import os
import time
import tempfile
import threading
import traceback
import concurrent.futures
import sys
from pathlib import Path

# --- Import shared library from utils subdir ---
try:
    from utils import llm_client
except ImportError:
    script_dir = Path(__file__).parent.resolve()
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from utils import llm_client

# Import specific functions/constants needed
from utils.llm_client import TRANSIENT_FAILURE_MARKER, VALID_PROVIDERS
from utils.llm_client import is_permanent_api_error

# --- Constants ---
PROGRESS_COUNTER = 0
PROGRESS_LOCK = threading.Lock()

# --- Helper Functions ---

def is_empty_content_response(resp):
    """
    Checks if a response object is structurally valid but has empty content.
    (Exact Logic from original ask.py)
    """
    try:
        if (isinstance(resp, dict) and
            "choices" in resp and
            isinstance(resp["choices"], list) and
            len(resp["choices"]) > 0 and
            isinstance(resp["choices"][0], dict) and
            resp["choices"][0].get("error") is None and
            isinstance(resp["choices"][0].get("message"), dict) and
            resp["choices"][0]["message"].get("content") == ""):
            return True
    except (TypeError, KeyError, IndexError):
        return False
    return False

# --- Loading and Cleanup Functions ---
def load_questions(file_path):
    """Load questions from a JSONL file. (Original Logic + Category handling)"""
    questions = []
    path = Path(file_path)
    if not path.is_file():
        print(f"Error: Input questions file not found: {file_path}", file=sys.stderr)
        return questions, False, None
    try:
        valid_count = 0
        skipped_count = 0
        first_category = None
        with path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    stripped_line = line.strip()
                    if stripped_line:
                        entry = json.loads(stripped_line)
                        if 'id' in entry and 'question' in entry:
                            entry['category'] = entry.get('category')
                            if not first_category and entry['category']:
                                first_category = entry['category']
                            questions.append(entry)
                            valid_count += 1
                        else:
                            print(f"Warning: Skipping line {i+1}: missing 'id' or 'question'.", file=sys.stderr)
                            skipped_count += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {i+1}: {e}.", file=sys.stderr)
                    skipped_count += 1
        print(f"Loaded {valid_count} valid questions from {file_path}.")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} invalid/malformed lines.")
        return questions, True, first_category
    except Exception as e:
        print(f"Error reading questions file {file_path}: {e}", file=sys.stderr)
        return questions, False, None

def load_existing_responses(output_file):
    """Load existing response IDs from output file. (Original Logic)"""
    processed_ids = set()
    lines_read = 0
    parse_warnings = 0
    if not output_file.exists():
        return processed_ids
    print(f"Loading existing responses from: {output_file}")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            lines_read += 1
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                entry = json.loads(stripped_line)
                question_id = entry.get("question_id")
                if question_id:
                    processed_ids.add(question_id)
                else:
                    # Original didn't print per line warning here
                    parse_warnings += 1
            except Exception as e:
                # Original just counted warnings on parse errors too
                print(f"Warning: Could not parse line {lines_read} or process entry: {str(e)}. Line: {stripped_line}", file=sys.stderr)
                parse_warnings += 1
    print(f"Finished loading. Lines processed: {lines_read}")
    print(f"  - Found {len(processed_ids)} logged entries (Successes OR Permanent Errors).")
    if parse_warnings > 0:
        print(f"  - Warnings during loading: {parse_warnings}")
    return processed_ids

def cleanup_permanent_errors(output_file):
    """
    Rewrites the output file, keeping ONLY successful responses.
    (Logic restored to match original ask.py, Syntax corrected)
    """
    input_path = Path(output_file)
    if not input_path.exists():
        print("Cleanup requested, but output file does not exist.")
        return True # Original returned True if no file

    temp_file_path = None
    lines_read, lines_written = 0, 0
    api_errors_skipped, empty_skipped, finish_reason_error_skipped, corrupted_skipped = 0, 0, 0, 0
    other_invalid_skipped = 0

    print(f"\n--- Cleaning up permanent errors in: {input_path} ---")
    print("--- Keeping ONLY successful responses (excluding finish_reason='error'). ---")
    success = True
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=input_path.parent, suffix='.tmp') as temp_f:
            temp_file_path = Path(temp_f.name)
            with open(input_path, 'r', encoding='utf-8') as original_f:
                for i, line in enumerate(original_f):
                    lines_read += 1
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue

                    keep_line = False
                    try:
                        entry = json.loads(stripped_line)
                        response_data = entry.get("response")
                        question_id = entry.get("question_id")

                        if not question_id or response_data is None:
                            corrupted_skipped += 1
                            continue

                        is_api_error = is_permanent_api_error(response_data)
                        is_empty = is_empty_content_response(response_data)
                        is_finish_reason_error = False
                        try:
                            if (isinstance(response_data, dict) and
                                response_data.get('choices') and
                                isinstance(response_data['choices'], list) and
                                len(response_data['choices']) > 0 and
                                isinstance(response_data['choices'][0], dict) and
                                response_data['choices'][0].get('finish_reason') == 'error'):
                                is_finish_reason_error = True
                        except (KeyError, IndexError, TypeError):
                            pass

                        is_valid_structure_for_keeping = False
                        try:
                             if (isinstance(response_data, dict) and
                                 "choices" in response_data and
                                 isinstance(response_data["choices"], list) and
                                 len(response_data["choices"]) > 0 and
                                 isinstance(response_data["choices"][0].get("message"), dict) and
                                 response_data["choices"][0]["message"].get("content") is not None):
                                 is_valid_structure_for_keeping = True
                        except (KeyError, IndexError, TypeError):
                           pass

                        if is_api_error:
                            api_errors_skipped += 1
                        elif is_empty:
                            empty_skipped += 1
                        elif is_finish_reason_error:
                            finish_reason_error_skipped += 1
                        elif not is_valid_structure_for_keeping:
                             other_invalid_skipped += 1
                        else:
                            keep_line = True

                        if keep_line:
                            temp_f.write(line)
                            lines_written += 1

                    except json.JSONDecodeError:
                        corrupted_skipped += 1
                        print(f"Warning: Skipping corrupted JSON line {lines_read} during cleanup.", file=sys.stderr)
                    except Exception as e_inner:
                         corrupted_skipped += 1
                         print(f"Warning: Error processing line {lines_read} during cleanup: {type(e_inner).__name__}. Skipping.", file=sys.stderr)
                         success = False

        # --- File Replacement Logic ---
        if lines_read > 0:
             print(f"\nCleanup processed {lines_read} lines.")
             print(f"  - Kept {lines_written} successful responses.")
             print(f"  - Skipped {api_errors_skipped} permanent API errors.")
             print(f"  - Skipped {empty_skipped} empty content responses.")
             print(f"  - Skipped {finish_reason_error_skipped} with finish_reason='error'.")
             combined_corrupted = corrupted_skipped + other_invalid_skipped
             print(f"  - Skipped {combined_corrupted} corrupted/unparseable/invalid structure lines.")

             changes_made = (lines_written != lines_read or combined_corrupted > 0)

             if success and changes_made :
                 if lines_written > 0 or api_errors_skipped > 0 or empty_skipped > 0 or finish_reason_error_skipped > 0 or combined_corrupted > 0 :
                     print("Attempting to atomically replace original file...")
                     try:
                         os.replace(temp_file_path, input_path)
                         print(f"Successfully replaced {input_path}.")
                         temp_file_path = None
                         print("--- Cleanup Finished Successfully ---")
                         return True
                     except OSError as e_replace:
                          print(f"!!! ERROR replacing file {input_path} with {temp_file_path}: {e_replace} !!!", file=sys.stderr)
                          return False
                 else:
                     print("No valid lines kept and no errors/empty found to remove.")
                     print("--- Cleanup Finished (No Changes Made) ---")
                     return True
             elif not success:
                 print("Cleanup process encountered errors. Original file not replaced.", file=sys.stderr)
                 return False
             else: # success and no changes made
                 print("No errors found to remove. Original file unchanged.")
                 print("--- Cleanup Finished (No Changes Made) ---")
                 return True
        else:
             print("Original file was empty or contained no processable lines. No cleanup performed.")
             return True

    except Exception as e:
        print(f"!!! ERROR during cleanup process: {type(e).__name__}: {e} !!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False
    finally:
        if temp_file_path and temp_file_path.exists():
            print(f"Cleaning up temporary file: {temp_file_path}")
            try:
                temp_file_path.unlink()
            except OSError as unlink_e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {unlink_e}", file=sys.stderr)


# --- API Interaction ---

def ask_question(question_text, question_id, model_id, api_target, api_key, openrouter_ignore_list=None):
    """
    Wrapper function calling the llm_client retry logic with ask.py's request maker.
    Passes the OpenRouter ignore list.
    """
    response = llm_client.retry_with_backoff(
        lambda: llm_client.make_api_call_ask(
            question_text=question_text, question_id=question_id, model_id=model_id,
            api_target=api_target, api_key=api_key,
            openrouter_ignore_list=openrouter_ignore_list
        ),
        context_info=f"QID {question_id}",
    )
    return response

# --- Worker Function ---
def process_single_question_worker(question_data, model_id, api_target, api_key, openrouter_ignore_list):
    """Worker function updated to pass ignore list."""
    question_id = question_data.get('id', '[Missing ID]')
    response = ask_question(
        question_data.get('question', '[Missing Question]'),
        question_id,
        model_id,
        api_target,
        api_key,
        openrouter_ignore_list=openrouter_ignore_list
        )
    return question_data, response

# --- Parameter Detection and Annotation ---
def guess_and_test_parameters(canonical_model_name):
    """Guesses provider and API model, tests using llm_client. (Restored Original Logic)"""
    print(f"Attempting to guess provider and API model for '{canonical_model_name}'...")
    if not isinstance(canonical_model_name, str) or not canonical_model_name:
        raise ValueError("Invalid canonical model name for guessing.")

    api_model_guess = canonical_model_name
    prefix_provider = None
    if '/' in canonical_model_name:
        api_model_guess = canonical_model_name.split('/')[-1]

    prefix_map = {"openai": "openai", "google": "google"}
    if '/' in canonical_model_name:
        prefix = canonical_model_name.split('/')[0].lower()
        prefix_provider = prefix_map.get(prefix)

    test_prompt = "Output the single word 'test' and nothing else."
    test_qid = "PARAM_TEST"

    # Attempt 1: Test OpenRouter
    provider_to_test_or = "openrouter"
    model_for_or_call = canonical_model_name
    print(f"  Attempt 1: Testing {provider_to_test_or.upper()} with model '{model_for_or_call}'...")
    or_key, or_env_var = llm_client.get_api_key_for_provider(provider_to_test_or, fail_if_missing=False)
    if or_key:
        try:
            test_response_or = ask_question(test_prompt, f"{test_qid}_OR", model_for_or_call, provider_to_test_or, or_key)
            is_or_success = False
            if test_response_or is not TRANSIENT_FAILURE_MARKER and not is_permanent_api_error(test_response_or):
                 try:
                     if (isinstance(test_response_or, dict) and
                         test_response_or.get('choices') and
                         isinstance(test_response_or['choices'], list) and
                         len(test_response_or['choices']) > 0 and
                         isinstance(test_response_or['choices'][0].get('message'), dict)):
                          is_or_success = True
                 except Exception:
                     pass # Ignore parsing errors during check
            if is_or_success:
                print(f"    Success! Using Provider='{provider_to_test_or}', API Model='{model_for_or_call}'")
                return provider_to_test_or, model_for_or_call
            else:
                err_info = "Transient Failure" if test_response_or is TRANSIENT_FAILURE_MARKER else str(test_response_or)[:200]
                print(f"    OpenRouter test failed. Response/Reason: {err_info}")
        except Exception as e:
            print(f"    OpenRouter test failed with exception: {e}")
    else:
        print(f"    Skipping OpenRouter test: Environment variable '{or_env_var}' not set.")

    # Attempt 2: Test Prefix Provider
    if prefix_provider and prefix_provider != "openrouter":
        provider_to_test_prefix = prefix_provider
        model_for_prefix_call = api_model_guess
        print(f"  Attempt 2: Testing {provider_to_test_prefix.upper()} with model '{model_for_prefix_call}'...")
        prefix_key, prefix_env_var = llm_client.get_api_key_for_provider(provider_to_test_prefix, fail_if_missing=True)
        try:
            test_response_prefix = ask_question(test_prompt, f"{test_qid}_PREFIX", model_for_prefix_call, provider_to_test_prefix, prefix_key)
            is_prefix_success = False
            if test_response_prefix is not TRANSIENT_FAILURE_MARKER and not is_permanent_api_error(test_response_prefix):
                 try:
                     if (isinstance(test_response_prefix, dict) and
                         test_response_prefix.get('choices') and
                         isinstance(test_response_prefix['choices'], list) and
                         len(test_response_prefix['choices']) > 0 and
                         isinstance(test_response_prefix['choices'][0].get('message'), dict)):
                          is_prefix_success = True
                 except Exception:
                     pass # Ignore parsing errors during check
            if is_prefix_success:
                print(f"    Success! Using Provider='{provider_to_test_prefix}', API Model='{model_for_prefix_call}'")
                return provider_to_test_prefix, model_for_prefix_call
            else:
                err_info = "Transient Failure" if test_response_prefix is TRANSIENT_FAILURE_MARKER else str(test_response_prefix)[:200]
                print(f"    Prefix provider ({provider_to_test_prefix}) test failed. Response/Reason: {err_info}")
        except Exception as e:
            print(f"    Prefix provider ({provider_to_test_prefix}) test failed with exception: {e}")
    elif prefix_provider == "openrouter":
        print("  Skipping Attempt 2: Prefix provider was OpenRouter (already tested).")
    else:
        print("  Skipping Attempt 2: No suitable prefix provider found or it was OpenRouter.")

    # Final Failure
    raise RuntimeError(f"Could not confirm working provider/model parameters for '{canonical_model_name}' after testing OpenRouter and potential prefix provider.")

def detect_and_verify_parameters(output_file_path, first_q_category=None):
    """Reads the first entry, detects params, guesses/tests if needed. (Restored Original Logic)"""
    print(f"Detecting parameters from first valid entry in: {output_file_path}")
    path = Path(output_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Output file for detection not found: {output_file_path}")

    detected_params = {
        'category': None, 'api_provider': None, 'api_model': None,
        'canonical_name': None, 'questions_file_path': None,
        'metadata_was_missing': False, 'output_file_path': str(output_file_path)
    }
    try:
        with path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue

                    entry = json.loads(stripped_line)
                    detected_params['category'] = entry.get('category')
                    if not detected_params['category']:
                         if first_q_category:
                             detected_params['category'] = first_q_category
                             print(f"Warning: Line {i+1} missing 'category', using fallback '{first_q_category}'.", file=sys.stderr)
                         else:
                             raise ValueError(f"First valid entry (line {i+1}) missing mandatory 'category' field.")

                    provider_raw = entry.get('api_provider')
                    detected_params['api_provider'] = provider_raw.lower() if provider_raw else None
                    detected_params['api_model'] = entry.get('api_model')
                    detected_params['canonical_name'] = entry.get('model')

                    if not detected_params['api_provider'] or not detected_params['api_model']:
                        print("  Metadata missing. Attempting guess and test...")
                        detected_params['metadata_was_missing'] = True
                        if not detected_params['canonical_name']:
                            raise ValueError(f"Cannot guess: 'model' field missing (line {i+1}).")
                        confirmed_provider, confirmed_api_model = guess_and_test_parameters(detected_params['canonical_name'])
                        detected_params['api_provider'] = confirmed_provider
                        detected_params['api_model'] = confirmed_api_model
                        print(f"  Guess successful: Provider='{confirmed_provider}', API Model='{confirmed_api_model}'")
                    else:
                         if detected_params['api_provider'] not in VALID_PROVIDERS:
                             print(f"Warning: Detected api_provider '{detected_params['api_provider']}' (line {i+1}) not in known providers {VALID_PROVIDERS}.", file=sys.stderr)
                         print(f"  Found existing metadata: Provider='{detected_params['api_provider']}', API Model='{detected_params['api_model']}'")

                    q_dir = Path("questions")
                    q_file = q_dir / f"{detected_params['category']}.jsonl"
                    detected_params['questions_file_path'] = str(q_file)
                    print(f"  Derived Questions File: {detected_params['questions_file_path']}")
                    return detected_params

                except json.JSONDecodeError:
                    continue
                except ValueError as ve:
                    raise ve
                except RuntimeError as rte:
                    raise rte
        raise ValueError(f"Could not find any valid JSON entries in {output_file_path} to detect parameters.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading {output_file_path} for detection: {e}") from e

def annotate_output_file(output_file_path, confirmed_provider, confirmed_api_model):
    """
    Rewrites the output file, adding/updating api_provider and api_model.
    (Syntax Corrected, Style Corrected)
    """
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
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=input_path.parent, suffix='.annot.tmp') as temp_f:
            temp_file_path = Path(temp_f.name)
            with input_path.open('r', encoding='utf-8') as infile:
                for i, line in enumerate(infile):
                    lines_read += 1
                    try:
                        entry = json.loads(line)
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
                        success = False

        if not success or error_count > 0:
            print(f"Annotation process encountered errors ({error_count}). Original file not replaced.", file=sys.stderr)
            success = False
        else:
            print(f"Annotation complete ({lines_written}/{lines_read} lines written). Replacing original file...")
            try:
                os.replace(temp_file_path, input_path)
                print(f"Successfully replaced {input_path} with annotated version.")
                temp_file_path = None
            except OSError as e_replace:
                print(f"!!! ERROR replacing file {input_path} with {temp_file_path}: {e_replace} !!!", file=sys.stderr)
                success = False

    except Exception as e:
        print(f"!!! CRITICAL ERROR during annotation process for {output_file_path}: {e} !!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        success = False
    finally:
        # Corrected Syntax and Style for finally block
        if temp_file_path and temp_file_path.exists():
            print(f"Cleaning up temporary annotation file: {temp_file_path}")
            try:
                temp_file_path.unlink()
            except OSError as unlink_e:
                print(f"Warning: Could not delete temporary annotation file {temp_file_path}: {unlink_e}", file=sys.stderr)
    return success

# --- Main Processing Logic ---
def process_questions(questions_file, model_id, canonical_name, api_target, api_key, num_workers,
                      openrouter_ignore_list, # Added ignore list
                      force_restart=False, force_retry_permanent=False, output_file_path_override=None, first_q_category=None):
    """
    Process questions in parallel, passing ignore list to worker.
    (Restored Original Logic, Corrected Style)
    """
    global PROGRESS_COUNTER
    PROGRESS_COUNTER = 0
    output_dir = Path("responses")
    output_dir.mkdir(exist_ok=True)

    # --- Determine Output File Path ---
    if output_file_path_override:
         output_file = Path(output_file_path_override)
         try:
             base_name = output_file.stem
             split_index = base_name.rfind('_')
             questions_filename = Path(questions_file).stem if questions_file else first_q_category or "unknown_category"
             safe_output_model_name = base_name[split_index+1:] if split_index != -1 else base_name
         except Exception:
             questions_filename = first_q_category or "unknown_category"
             safe_output_model_name = "unknown_model"
         print(f"Using specified output file: {output_file}")
    else:
         questions_filename = Path(questions_file).stem
         output_model_name = canonical_name if canonical_name else model_id
         safe_output_model_name = output_model_name.replace('/', '_').replace(':', '-')
         output_file = output_dir / f"{questions_filename}_{safe_output_model_name}.jsonl"
         print(f"Using constructed output file: {output_file}")

    # --- Load Questions ---
    questions, loaded_ok, _ = load_questions(questions_file)
    if not loaded_ok:
        raise IOError(f"Failed to load questions from {questions_file}")
    total_questions = len(questions)

    # --- Load Existing Responses / Handle Restart ---
    processed_ids = set()
    if output_file.exists() and not force_restart:
        processed_ids = load_existing_responses(output_file)
        if force_retry_permanent:
            print(f"Resuming run with FRPE active. Found {len(processed_ids)} entries remaining after cleanup.", file=sys.stderr)
        else:
            print(f"Resuming run with FRPE inactive. Found {len(processed_ids)} logged entries.", file=sys.stderr)
    elif not output_file.exists():
        print(f"No existing output file found at '{output_file}'. Starting fresh.")

    # --- Filter Questions ---
    questions_to_process = []
    for q_data in questions:
        q_id = q_data.get('id')
        if not q_id:
            print(f"Warning: Skipping question missing 'id': {q_data.get('question', '')[:50]}...", file=sys.stderr)
            continue
        if q_id not in processed_ids:
            questions_to_process.append(q_data)

    num_to_process = len(questions_to_process)
    num_skipped = total_questions - num_to_process
    print(f"\nTotal questions in source file: {total_questions}")
    print(f"Already processed/logged: {num_skipped}")
    print(f"Questions to process this run: {num_to_process}")
    if num_to_process == 0:
        print("No new questions found to process for this configuration.")
        return

    # --- Prepare for Execution ---
    model_name_to_log = canonical_name if canonical_name else model_id
    print(f"\nProcessing {num_to_process} questions for model '{model_name_to_log}' via '{api_target.upper()}' ({num_workers} workers)...")
    if api_target == 'openrouter' and openrouter_ignore_list:
        print(f"  Ignoring OpenRouter providers: {openrouter_ignore_list}")
    if force_retry_permanent:
        print(f"Saving ALL responses (Successes and Retried Permanent Errors) to: {output_file}")
    else:
        print(f"Saving responses for NEW questions (Successes and Permanent Errors) to: {output_file}")

    output_lock = threading.Lock()
    questions_processed_this_run = 0
    questions_logged_this_run = 0
    questions_failed_transiently_this_run = 0
    start_time = time.time()

    # --- Thread Pool Execution ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_question_worker,
                                   q_data, model_id, api_target, api_key,
                                   openrouter_ignore_list)
                   for q_data in questions_to_process]

        for future in concurrent.futures.as_completed(futures):
            try:
                question_data, response_dict = future.result()
                question_id = question_data.get('id', '[Missing ID]')
                questions_processed_this_run += 1

                if response_dict is TRANSIENT_FAILURE_MARKER:
                    questions_failed_transiently_this_run += 1
                else:
                    provider_used = response_dict.pop("_provider_used", api_target.lower())
                    model_used_for_api = response_dict.pop("_model_used_for_api", model_id)
                    raw_provider_response = response_dict.pop("_raw_provider_response", None)

                    # Construct Output Entry (Original Format)
                    output_entry = {
                        "question_id": question_id,
                        "category": question_data.get('category', first_q_category),
                        "question": question_data.get('question'),
                        "model": model_name_to_log,
                        "api_provider": provider_used,
                        "api_model": model_used_for_api,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "response": response_dict
                    }
                    if raw_provider_response is not None:
                        output_entry["raw_response"] = raw_provider_response
                    if 'domain' in question_data:
                        output_entry['domain'] = question_data['domain']

                    # Log the entry
                    try:
                        with output_lock:
                            with open(output_file, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
                        questions_logged_this_run += 1

                        # Update Progress
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
                        # Original continued processing

            except Exception as future_e:
                 print(f"\nCRITICAL ERROR processing future result: {type(future_e).__name__}: {future_e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)
                 # Original continued processing

    # Final Summary (Original Logic)
    end_time = time.time()
    duration = end_time - start_time
    print("-" * 30)
    print(f"Finished processing questions from {questions_file}.")
    print(f"  Output file: {output_file}")
    print(f"  Model requested for API: {model_id}")
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
    try:
        final_processed_ids = load_existing_responses(output_file)
        print(f"  Total logged entries in file now: {len(final_processed_ids)} / {total_questions}")
    except Exception as e:
        print(f"  Warning: Could not reload final count from output file: {e}")
    print("-" * 30)

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description='Ask questions via API in parallel, reprocess existing outputs, or test model coherency.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--normal', action='store_true', help='Normal mode: Run questions from input files.')
    mode_group.add_argument('--detect', metavar='OUTPUT_FILE', help='Detect mode: Process an existing output file, detecting parameters. Use with --frpe to retry errors, or without to only process new questions.')
    parser.add_argument('--model', help='[Normal Mode / Override in Detect Mode] Model identifier specific to the chosen API provider (e.g., gpt-4o, google/gemini-pro).')
    parser.add_argument('--questions_files', nargs='+', help='[Normal Mode] One or more JSONL question files (must contain id, question). Category is recommended.')
    parser.add_argument('--provider', choices=VALID_PROVIDERS, help='[Normal Mode Required / Override in Detect Mode] Specify the API provider.')
    parser.add_argument('--canonical-name', help='[Optional in Both Modes] Canonical name for model in output/logs (e.g., openai/gpt-4o). Overrides detected/inferred.')
    parser.add_argument('-w', '--workers', type=int, default=4, help='Number of parallel worker threads.')
    parser.add_argument('--force-restart', action='store_true', help='Delete existing output file(s) and start fresh (runs before FRPE cleanup).')
    parser.add_argument('--frpe', '--force-retry-permanent-errors', dest='frpe', action='store_true', help='[Optional] Clean permanent errors/empty responses from output file before processing, forcing retry.')
    parser.add_argument('--test', action='store_true', help='Run basic coherency tests on the target model before processing questions. Exits if tests fail. Requires OPENROUTER_API_KEY for judge.')
    args = parser.parse_args()

    # Argument validation
    if args.normal:
        if not args.model or not args.questions_files or not args.provider:
            parser.error("--normal mode requires --model, --questions_files, and --provider.")
        if args.detect:
            parser.error("Cannot use --detect with --normal.")
        if args.frpe:
            print("Warning: Using --frpe in --normal mode will clean the target output file(s) before writing new results.")
    elif args.detect:
        if args.questions_files:
            parser.error("Cannot specify --questions_files in --detect mode (derived from category).")
    if args.workers < 1:
        parser.error("Number of workers must be at least 1.")
    if args.force_restart and args.frpe:
        print("Note: --force-restart is active; output file will be deleted before processing, skipping specific FRPE cleanup step.")

    # --- Main Logic ---
    overall_success = True
    openrouter_ignore_list = [] # Initialize ignore list

    try:
        if args.normal:
            # --- Normal Mode ---
            print("Running in NORMAL mode.")
            run_api_target = args.provider.lower()
            run_model_id = args.model
            run_canonical_name = args.canonical_name if args.canonical_name else None
            run_api_key, run_env_var_used = llm_client.get_api_key_for_provider(run_api_target)
            log_final_config(run_api_target, run_env_var_used, args.workers, run_model_id, run_canonical_name)

            # --- Optional Coherency Test ---
            if args.test:
                test_passed, failed_providers = llm_client.run_coherency_tests(run_model_id, run_api_target, run_api_key)
                if not test_passed:
                    print("!!! FATAL: Coherency tests failed for the model. Exiting. !!!", file=sys.stderr)
                    sys.exit(1)
                else:
                    openrouter_ignore_list = failed_providers
                    print("--- Coherency tests passed. Proceeding with question processing. ---")

            total_start_time = time.time()
            files_processed_count = 0
            files_failed_count = 0
            first_category_overall = None

            for questions_file in args.questions_files:
                 print(f"\n=== Starting processing for file: {questions_file} ===")
                 file_start_time = time.time()
                 output_file_to_process = None
                 current_file_category = None
                 try:
                     _, temp_loaded_ok, temp_first_cat = load_questions(questions_file)
                     if not temp_loaded_ok:
                         raise IOError(f"Failed to load questions from {questions_file}")
                     current_file_category = temp_first_cat
                     if not first_category_overall:
                         first_category_overall = current_file_category

                     output_dir = Path("responses")
                     output_dir.mkdir(exist_ok=True)
                     questions_filename_stem = Path(questions_file).stem
                     output_model_name_part = run_canonical_name if run_canonical_name else run_model_id
                     safe_output_model_name = output_model_name_part.replace('/', '_').replace(':', '-')
                     output_file_to_process = output_dir / f"{questions_filename_stem}_{safe_output_model_name}.jsonl"

                     if args.force_restart and output_file_to_process.exists():
                          print(f"Force restart: Deleting existing output file '{output_file_to_process}'")
                          try:
                              output_file_to_process.unlink()
                          except OSError as e:
                              print(f"Error deleting {output_file_to_process}: {e}.", file=sys.stderr)
                              raise IOError(f"Delete failed for {output_file_to_process}") from e

                     if args.frpe and not args.force_restart:
                          if not cleanup_permanent_errors(output_file_to_process):
                              print(f"Error: FRPE cleanup failed for {output_file_to_process}. Skipping.", file=sys.stderr)
                              raise RuntimeError("FRPE Cleanup Failed")

                     process_questions(
                         questions_file=questions_file, model_id=run_model_id, canonical_name=run_canonical_name,
                         api_target=run_api_target, api_key=run_api_key, num_workers=args.workers,
                         openrouter_ignore_list=openrouter_ignore_list, # Pass list
                         force_restart=args.force_restart, force_retry_permanent=args.frpe,
                         output_file_path_override=None, first_q_category=current_file_category
                     )
                     files_processed_count += 1

                 except KeyboardInterrupt:
                     raise
                 except (IOError, FileNotFoundError, RuntimeError) as e:
                     files_failed_count += 1
                     print(f"Error preparing or processing {questions_file}: {e}", file=sys.stderr)
                     overall_success = False
                 except Exception as e:
                     files_failed_count += 1
                     print(f"\n!!! UNEXPECTED ERROR processing {questions_file}: {type(e).__name__}: {str(e)} !!!", file=sys.stderr)
                     traceback.print_exc(file=sys.stderr)
                     overall_success = False

                 file_end_time = time.time()
                 print(f"=== Finished processing file: {questions_file} (Duration: {file_end_time - file_start_time:.2f} seconds) ===")

            # Normal Mode Summary
            total_end_time = time.time()
            print("\n" + "="*40)
            print("Overall Summary (Normal Mode):")
            print(f"  API Model used: {run_model_id}")
            if run_canonical_name:
                print(f"  Canonical Name logged: {run_canonical_name}")
            print(f"  API Provider used: {run_api_target.upper()}")
            print(f"  Total files attempted: {len(args.questions_files)}")
            print(f"  Files processed successfully: {files_processed_count}")
            print(f"  Files failed/skipped: {files_failed_count}")
            print(f"  Total execution time: {total_end_time - total_start_time:.2f} seconds")
            print("="*40)

        elif args.detect:
            # --- Detect Mode ---
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

            q_cat_fallback = Path(output_file_to_detect).stem.split('_')[0] if '_' in Path(output_file_to_detect).stem else None
            detected_params = detect_and_verify_parameters(output_file_to_detect, first_q_category=q_cat_fallback)

            if detected_params['metadata_was_missing']:
                if output_file_path.exists():
                    if not annotate_output_file(output_file_to_detect, detected_params['api_provider'], detected_params['api_model']):
                        print(f"FATAL ERROR: Failed to annotate output file {output_file_to_detect}.", file=sys.stderr)
                        sys.exit(1)
                else:
                    print("Note: Annotation skipped as file was deleted by --force-restart.")

            final_api_target = args.provider.lower() if args.provider else detected_params['api_provider']
            final_api_model = args.model if args.model else detected_params['api_model']
            final_canonical_name = args.canonical_name if args.canonical_name else detected_params['canonical_name']
            questions_file_path = detected_params['questions_file_path']

            if not questions_file_path:
                print(f"FATAL ERROR: Could not determine questions file path from category '{detected_params['category']}'.", file=sys.stderr)
                sys.exit(1)
            if not Path(questions_file_path).is_file():
                print(f"FATAL ERROR: Derived questions file path '{questions_file_path}' does not exist.", file=sys.stderr)
                sys.exit(1)
            if not final_api_target or not final_api_model:
                 print(f"FATAL ERROR: Could not determine final API Provider ('{final_api_target}') or Model ('{final_api_model}').", file=sys.stderr)
                 sys.exit(1)

            final_api_key, final_env_var_used = llm_client.get_api_key_for_provider(final_api_target)
            log_final_config(final_api_target, final_env_var_used, args.workers, final_api_model, final_canonical_name, detected_params)

            # --- Optional Coherency Test ---
            if args.test:
                test_passed, failed_providers = llm_client.run_coherency_tests(final_api_model, final_api_target, final_api_key)
                if not test_passed:
                    print("!!! FATAL: Coherency tests failed for the model. Exiting. !!!", file=sys.stderr)
                    sys.exit(1)
                else:
                    openrouter_ignore_list = failed_providers
                    print("--- Coherency tests passed. Proceeding with question processing. ---")

            # Handle FRPE cleanup
            if args.frpe and not args.force_restart:
                print(f"\nRunning FRPE cleanup on: {output_file_to_detect}")
                if not cleanup_permanent_errors(output_file_to_detect):
                    print(f"FATAL ERROR: Cleanup failed for {output_file_to_detect}.", file=sys.stderr)
                    sys.exit(1)
            elif args.frpe and args.force_restart:
                print("\nSkipping FRPE cleanup because --force-restart was used.")
            elif not args.frpe:
                print("\n--frpe not specified, skipping cleanup of existing errors.")

            # Process questions
            print(f"\n=== Starting processing for detected file: {output_file_to_detect} ===")
            process_questions(
                questions_file=questions_file_path, model_id=final_api_model, canonical_name=final_canonical_name,
                api_target=final_api_target, api_key=final_api_key, num_workers=args.workers,
                openrouter_ignore_list=openrouter_ignore_list, # Pass list
                force_restart=False, # Restart handled above
                force_retry_permanent=args.frpe,
                output_file_path_override=output_file_to_detect,
                first_q_category=detected_params['category']
            )

    # --- Global Error Handling ---
    except (ValueError, RuntimeError, IOError, FileNotFoundError) as e:
        print(f"\n!!! SETUP/CONFIGURATION ERROR: {type(e).__name__}: {str(e)} !!!", file=sys.stderr)
        overall_success = False
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        overall_success = False
    except Exception as e:
        print(f"\n!!! CRITICAL UNEXPECTED ERROR: {type(e).__name__}: {str(e)} !!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        overall_success = False

    sys.exit(0 if overall_success else 1)

# --- Logging Helper ---
def log_final_config(api_target, env_var_used, workers, model_id, canonical_name, detected_params=None):
    """Logs the final configuration parameters being used. (Original Logic)"""
    # Unchanged
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
