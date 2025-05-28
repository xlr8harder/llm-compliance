#!/usr/bin/env python3
"""Check for duplicate question_id entries in JSONL analysis files."""

import json
import sys
from collections import defaultdict
from pathlib import Path

def check_file_for_duplicates(filepath):
    """Check a single JSONL file for duplicate question_ids."""
    question_ids = defaultdict(list)
    
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    qid = data.get('question_id')
                    if qid:
                        question_ids[qid].append({
                            'line': line_num,
                            'timestamp': data.get('timestamp', 'N/A'),
                            'compliance': data.get('compliance', 'N/A')
                        })
                except json.JSONDecodeError as e:
                    print(f"  JSON error on line {line_num}: {e}")
                    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}
    
    return question_ids

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_duplicates.py <file_or_directory> [file2] [file3] ...")
        print("  If directory provided, checks all .jsonl files in it")
        sys.exit(1)
    
    files_to_check = []
    
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_file():
            files_to_check.append(path)
        elif path.is_dir():
            files_to_check.extend(path.glob("*.jsonl"))
        else:
            print(f"Warning: {arg} is not a valid file or directory")
    
    if not files_to_check:
        print("No JSONL files found to check")
        sys.exit(1)
    
    total_duplicates = 0
    
    for filepath in sorted(files_to_check):
        question_ids = check_file_for_duplicates(filepath)
        
        duplicates = {qid: entries for qid, entries in question_ids.items() if len(entries) > 1}
        
        if duplicates:
            print(f"\nChecking: {filepath}")
            print(f"  Found {len(duplicates)} question_ids with duplicates:")
            for qid, entries in duplicates.items():
                print(f"    {qid}: {len(entries)} entries")
                for entry in entries:
                    print(f"      Line {entry['line']}: {entry['compliance']} @ {entry['timestamp']}")
            total_duplicates += len(duplicates)
    
    print(f"\n=== SUMMARY ===")
    print(f"Checked {len(files_to_check)} files")
    print(f"Total question_ids with duplicates: {total_duplicates}")

if __name__ == "__main__":
    main()
