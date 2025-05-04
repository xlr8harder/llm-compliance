import sys
import json

def add_model_field(input_file, output_file, model_name):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip():  # skip empty lines
                obj = json.loads(line)
                obj['model'] = model_name
                outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python add_model_field.py input.jsonl output.jsonl model-name")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model_name = sys.argv[3]

    add_model_field(input_path, output_path, model_name)
