import argparse
import json
import os
import tempfile
import shutil

def add_field_to_jsonl_in_place(input_file, field_name, field_value):
    dir_name = os.path.dirname(input_file)
    base_name = os.path.basename(input_file)

    # Create a temporary file in the same directory
    with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, encoding='utf-8') as tmpfile:
        temp_path = tmpfile.name
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                if line.strip():
                    obj = json.loads(line)
                    obj[field_name] = field_value
                    tmpfile.write(json.dumps(obj, ensure_ascii=False) + '\n')

    # Atomically replace the original file
    shutil.move(temp_path, input_file)

def main():
    parser = argparse.ArgumentParser(description='Add or overwrite a field with a fixed value in-place in a JSONL file.')
    parser.add_argument('-f', '--field', required=True, help='Field name to add or overwrite')
    parser.add_argument('-v', '--value', required=True, help='Value to set for the field')
    parser.add_argument('-i', '--input', required=True, help='Input JSONL file (will be modified in-place)')

    args = parser.parse_args()

    add_field_to_jsonl_in_place(args.input, args.field, args.value)

if __name__ == '__main__':
    main()
