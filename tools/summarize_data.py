#!/usr/bin/env python3
"""
Script to summarize the contents of compliance data files.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple, Any

# Add the parent directory to the path so we can import compliance if needed
script_dir = Path(__file__).parent.resolve()
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


def load_jsonl(file_path):
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
    return data

def count_field_values(data, field_name):
    """Count occurrences of values for a specific field."""
    counter = Counter()
    for item in data:
        if field_name in item:
            value = item[field_name]
            # Handle dictionary values
            if isinstance(value, dict):
                counter["[dict]"] += 1
            # Handle list values
            elif isinstance(value, list):
                counter["[list]"] += 1
            # Count actual values
            else:
                counter[value] += 1
        else:
            counter["[missing]"] += 1
    return counter

def analyze_field_structure(data, field_name):
    """Analyze the structure of a field that contains dictionaries."""
    # Collect all keys used in the field
    all_keys = set()
    for item in data:
        if field_name in item and isinstance(item[field_name], dict):
            all_keys.update(item[field_name].keys())
    
    # Count occurrences of each key
    key_counts = defaultdict(int)
    for item in data:
        if field_name in item and isinstance(item[field_name], dict):
            for key in all_keys:
                if key in item[field_name]:
                    key_counts[key] += 1
    
    return key_counts, all_keys

def find_all_fields(data):
    """Find all fields used across all records."""
    all_fields = set()
    for item in data:
        all_fields.update(item.keys())
    return all_fields

def count_field_presence(data, all_fields):
    """Count how many records have each field."""
    field_counts = {field: 0 for field in all_fields}
    for item in data:
        for field in item.keys():
            field_counts[field] += 1
    return field_counts

def summarize_file(file_path):
    """Generate a summary of a JSONL file."""
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return
    
    print(f"\n=== Summary for {path} ===\n")
    
    # Load the data
    data = load_jsonl(path)
    if not data:
        print("File is empty or contains no valid JSON records")
        return
    
    # Basic info
    print(f"Total records: {len(data)}")
    
    # Find all fields
    all_fields = find_all_fields(data)
    print(f"Total unique fields: {len(all_fields)}")
    
    # Count field presence
    field_counts = count_field_presence(data, all_fields)
    print("\nField presence:")
    for field, count in sorted(field_counts.items(), key=lambda x: (-x[1], x[0])):
        percentage = (count / len(data)) * 100
        print(f"  {field}: {count} ({percentage:.1f}%)")
    
    # Analyze key fields based on file type
    if any("question_id" in item for item in data):
        # Response or analysis file
        if any("compliance" in item for item in data):
            # Analysis file
            print("\nCompliance ratings:")
            compliance_counts = count_field_values(data, "compliance")
            for value, count in compliance_counts.most_common():
                percentage = (count / len(data)) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
            
            print("\nJudge models used:")
            judge_model_counts = count_field_values(data, "judge_model")
            for value, count in judge_model_counts.most_common():
                percentage = (count / len(data)) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
                
            print("\nJudge providers used:")
            judge_provider_counts = count_field_values(data, "judge_api_provider")
            for value, count in judge_provider_counts.most_common():
                percentage = (count / len(data)) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
        else:
            # Response file
            print("\nModels used:")
            model_counts = count_field_values(data, "model")
            for value, count in model_counts.most_common():
                percentage = (count / len(data)) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
                
            print("\nProviders used:")
            provider_counts = count_field_values(data, "api_provider")
            for value, count in provider_counts.most_common():
                percentage = (count / len(data)) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
    else:
        # Question file
        print("\nQuestion categories:")
        category_counts = count_field_values(data, "category")
        for value, count in category_counts.most_common():
            percentage = (count / len(data)) * 100
            print(f"  {value}: {count} ({percentage:.1f}%)")
            
        print("\nQuestion domains:")
        domain_counts = count_field_values(data, "domain")
        for value, count in domain_counts.most_common():
            percentage = (count / len(data)) * 100
            print(f"  {value}: {count} ({percentage:.1f}%)")
    
    # Analyze complex fields
    print("\nComplex field analysis:")
    
    if any("response" in item for item in data):
        print("\nResponse field structure:")
        response_keys, all_response_keys = analyze_field_structure(data, "response")
        for key, count in sorted(response_keys.items(), key=lambda x: (-x[1], x[0])):
            percentage = (count / len(data)) * 100
            print(f"  {key}: {count} ({percentage:.1f}%)")
            
    if any("raw_response" in item for item in data):
        print("\nRaw response field structure:")
        raw_keys, all_raw_keys = analyze_field_structure(data, "raw_response")
        for key, count in sorted(raw_keys.items(), key=lambda x: (-x[1], x[0])):
            percentage = (count / len(data)) * 100
            print(f"  {key}: {count} ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Summarize compliance data files")
    parser.add_argument("file", help="Path to JSONL file to summarize")
    
    args = parser.parse_args()
    summarize_file(args.file)

if __name__ == "__main__":
    main()
