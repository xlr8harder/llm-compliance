# post_process_judge_errors.py
import argparse
import json
import os
import sys
import time
import tempfile
import traceback
from pathlib import Path

def process_file(filepath):
    """
    Processes a single judge output file (.jsonl).
    Corrects entries where the original response had finish_reason 'error'
    but the compliance was not marked as ERROR_ORIGINAL_RESPONSE.
    Writes changes to a temporary file and replaces the original atomically.
    """
    input_path = Path(filepath)
    if not input_path.is_file():
        print(f"Error: Input file not found: {filepath}", file=sys.stderr)
        return False

    temp_file_path = None
    modified_count = 0
    lines_read = 0
    error_count = 0
    success = True

    print(f"\n--- Processing file: {filepath} ---")

    try:
        # Create temp file in the same directory for atomic rename
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False,
                                         dir=input_path.parent, suffix='.tmp') as temp_f:
            temp_file_path = Path(temp_f.name)
            with input_path.open('r', encoding='utf-8') as infile:
                for i, line in enumerate(infile):
                    lines_read = i + 1
                    try:
                        entry = json.loads(line)
                        needs_correction = False

                        # Safely check for the condition:
                        # response exists -> choices[0] exists -> finish_reason is "error"
                        # AND compliance is not already an error state
                        original_response = entry.get('response', {})
                        choices = original_response.get('choices', [])
                        finish_reason = None
                        if isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], dict):
                            finish_reason = choices[0].get('finish_reason')

                        current_compliance = entry.get('compliance')

                        if finish_reason == "error" and \
                           isinstance(current_compliance, str) and \
                           not current_compliance.startswith("ERROR_"):
                            needs_correction = True

                        if needs_correction:
                            original_compliance = entry['compliance']
                            original_analysis = entry.get('judge_analysis', '')

                            entry['compliance'] = "ERROR_ORIGINAL_RESPONSE"
                            entry['judge_analysis'] = f"Post-processed: Compliance changed from '{original_compliance}' to 'ERROR_ORIGINAL_RESPONSE' because original response had finish_reason='error'. Original Analysis: {original_analysis}"
                            modified_count += 1
                            print(f"  Corrected QID {entry.get('question_id', 'UNKNOWN')}: Was '{original_compliance}', now 'ERROR_ORIGINAL_RESPONSE'")

                        # Write original or modified entry to temp file
                        temp_f.write(json.dumps(entry, ensure_ascii=False) + '\n')

                    except json.JSONDecodeError:
                        print(f"Warning: Skipping corrupted JSON line {lines_read}", file=sys.stderr)
                        # Write the corrupted line as-is to preserve it? Or skip? Skipping for now.
                        # temp_f.write(line) # Uncomment to preserve corrupted lines
                        error_count += 1
                        continue # Skip processing this line further
                    except Exception as e:
                        print(f"Error processing line {lines_read}: {e}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
                        # Write original line to temp file on unexpected error to avoid data loss
                        temp_f.write(line)
                        error_count += 1
                        success = False # Mark file processing as failed


        # --- Post-processing and replacement ---
        if error_count > 0:
             print(f"Encountered {error_count} errors processing lines.")
             # Decide if we should still replace if errors occurred but some lines were processed
             # For safety, let's NOT replace if errors occurred during processing lines
             print("Replacement skipped due to processing errors.", file=sys.stderr)
             success = False

        elif modified_count > 0:
            print(f"  {modified_count} entries corrected.")
            print(f"Attempting to atomically replace original file '{input_path}'...")
            try:
                # tempfile is closed, safe to replace
                os.replace(temp_file_path, input_path)
                print(f"Successfully replaced {input_path}.")
                temp_file_path = None # Prevent deletion in finally block
            except OSError as e_replace:
                print(f"!!! ERROR replacing file {input_path} with {temp_file_path}: {e_replace} !!!", file=sys.stderr)
                success = False
        else:
            print("  No corrections needed for this file.")
            # No need to replace, temp file will be removed

    except Exception as e:
        print(f"!!! CRITICAL ERROR processing file {filepath}: {e} !!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        success = False
    finally:
        # Clean up temp file if replacement failed or wasn't needed/attempted
        if temp_file_path and temp_file_path.exists():
            print(f"Cleaning up temporary file: {temp_file_path}")
            try:
                temp_file_path.unlink()
            except OSError as unlink_e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {unlink_e}", file=sys.stderr)

    print(f"--- Finished processing: {filepath} {'(Success)' if success else '(Failed)'} ---")
    return success


def main():
    parser = argparse.ArgumentParser(
        description='Post-process judge_compliance output files (.jsonl) to correct entries where the original response had finish_reason="error" but compliance was not marked as an error.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('judge_output_files', nargs='+', help='One or more judge output JSONL files to process.')

    args = parser.parse_args()

    total_files = len(args.judge_output_files)
    files_processed_successfully = 0
    files_failed = 0
    overall_start_time = time.time()

    for filepath in args.judge_output_files:
        if process_file(filepath):
            files_processed_successfully += 1
        else:
            files_failed += 1

    overall_end_time = time.time()
    print("\n" + "="*40)
    print("Overall Post-Processing Summary:")
    print(f"  Total files attempted: {total_files}")
    print(f"  Files processed successfully: {files_processed_successfully}")
    print(f"  Files failed processing: {files_failed}")
    print(f"  Total execution time: {overall_end_time - overall_start_time:.2f} seconds")
    print("="*40)

    if files_failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
