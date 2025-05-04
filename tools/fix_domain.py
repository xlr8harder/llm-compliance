import json
import argparse
import os
import tempfile
import shutil

def load_domain_map(good_file_path):
    """Build a map of question_id -> domain from the known-good file."""
    domain_map = {}
    with open(good_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                qid = obj.get("question_id")
                domain = obj.get("domain")
                if qid and domain:
                    domain_map[qid] = domain
    return domain_map

def transplant_domains(bad_file_path, domain_map):
    """Transplant domains into a file missing them and overwrite atomically."""
    dir_name = os.path.dirname(bad_file_path)

    with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, encoding='utf-8') as tmpfile:
        temp_path = tmpfile.name
        with open(bad_file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                if line.strip():
                    obj = json.loads(line)
                    qid = obj.get("question_id")
                    if qid and "domain" not in obj:
                        if qid in domain_map:
                            obj["domain"] = domain_map[qid]
                        else:
                            print(f"Warning: No domain found for question_id: {qid}")
                    tmpfile.write(json.dumps(obj, ensure_ascii=False) + '\n')

    os.rename(temp_path, bad_file_path)

def main():
    parser = argparse.ArgumentParser(description='Repair missing "domain" fields in a JSONL file using a reference file.')
    parser.add_argument('--good', '-g', required=True, help='Path to known-good JSONL file (with domains)')
    parser.add_argument('--bad', '-b', required=True, help='Path to JSONL file missing domains (will be modified in place)')

    args = parser.parse_args()

    domain_map = load_domain_map(args.good)
    transplant_domains(args.bad, domain_map)
    print(f"✅ Transplanted domain fields into: {args.bad}")

if __name__ == '__main__':
    main()

