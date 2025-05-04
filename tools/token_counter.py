import json
import sys
from pathlib import Path

def total_tokens_from_files(file_paths):
    grand_prompt_tokens = 0
    grand_completion_tokens = 0

    print("Per-file summary:\n")

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"{file_path}: File not found.\n")
            continue

        file_prompt_tokens = 0
        file_completion_tokens = 0

        with path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    usage = data.get("response", {}).get("usage", {})
                    file_prompt_tokens += usage.get("prompt_tokens", 0)
                    file_completion_tokens += usage.get("completion_tokens", 0)
                except json.JSONDecodeError as e:
                    print(f"{file_path}: Skipping invalid line: {e}")

        file_total = file_prompt_tokens + file_completion_tokens
        grand_prompt_tokens += file_prompt_tokens
        grand_completion_tokens += file_completion_tokens

        print(f"{file_path}")
        print(f"  Prompt tokens:     {file_prompt_tokens}")
        print(f"  Completion tokens: {file_completion_tokens}")
        print(f"  Total tokens:      {file_total}\n")

    grand_total = grand_prompt_tokens + grand_completion_tokens
    print("Grand total:")
    print(f"  Prompt tokens:     {grand_prompt_tokens}")
    print(f"  Completion tokens: {grand_completion_tokens}")
    print(f"  Total tokens:      {grand_total}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python token_counter.py <file1.jsonl> [file2.jsonl ...]")
        sys.exit(1)

    total_tokens_from_files(sys.argv[1:])
