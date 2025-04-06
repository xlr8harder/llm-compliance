import argparse
import datetime
import json
import os
import requests
import time
from pathlib import Path
from requests.exceptions import RequestException

def load_questions(file_path):
    """Load questions from a JSONL file."""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    return questions

def load_existing_responses(output_file):
    """Load existing responses from output file and return a set of processed question IDs."""
    processed_ids = set()
    
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    processed_ids.add(entry["question_id"])
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not parse line in output file: {str(e)}")
    
    return processed_ids

def retry_with_backoff(func, max_retries=8, initial_delay=1):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
    """
    retries = 0
    while retries < max_retries:
        try:
            return func()
        except (RequestException, ValueError, json.JSONDecodeError) as e:
            print(e)
            retries += 1
            if retries == max_retries:
                return {"error": f"After {max_retries} retries: {str(e)}"}

            delay = initial_delay * (2 ** (retries - 1))
            print(f"Attempt {retries} failed with error: {str(e)}. Retrying in {delay} seconds...")
            time.sleep(delay)

    raise ValueError("No valid responses received.")

def ask_question(question, model, api_key):
    """Send a question to the appropriate API endpoint based on model name."""
    # Determine API endpoint and key based on model name
    if model.startswith("accounts/fireworks"):
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }
    
    def make_request():
        response = requests.post(
            url=url,
            headers=headers,
            data=json.dumps(data),
            timeout=30
        )
        response_data = response.json()

        if response.status_code >= 500 or (isinstance(response_data, dict) and 'error' in response_data):
            raise RequestException(f"Server error: {response_data.get('error', {}).get('message', 'Unknown error')}")
        elif not response_data["choices"][0]["message"]["content"]:
            raise RequestException(f"Received empty response from server")
        return response_data
    
    return retry_with_backoff(make_request)

def process_questions(questions_file, model, api_key, force_restart=False):
    """Process all questions in a file and save responses. Resume if output file exists."""
    output_dir = Path("responses")
    output_dir.mkdir(exist_ok=True)
    
    questions_filename = Path(questions_file).stem
    output_file = output_dir / f"{questions_filename}_{model.replace('/', '_')}.jsonl"
    
    # Load all questions from input file
    questions = load_questions(questions_file)
    total_questions = len(questions)
    
    # Skip if output file exists and has the correct number of lines (unless force_restart is True)
    if output_file.exists() and not force_restart:
        processed_ids = load_existing_responses(output_file)
        
        if len(processed_ids) == total_questions:
            print(f"Output file '{output_file}' already contains all {total_questions} responses. Skipping.")
            return
        elif len(processed_ids) > 0:
            print(f"Found {len(processed_ids)}/{total_questions} responses in existing output file. Resuming from where we left off.")
    else:
        processed_ids = set()
        if force_restart and output_file.exists():
            print(f"Force restart requested. Removing existing output file '{output_file}'.")
            output_file.unlink()
    
    api_type = "Fireworks" if model.startswith("accounts/fireworks") else "OpenRouter"
    print(f"Processing questions from {questions_file}")
    print(f"Using {api_type} API")
    
    for i, question_data in enumerate(questions, 1):
        question_id = question_data['id']
        
        # Skip questions that have already been processed
        if question_id in processed_ids:
            continue
        
        print(f"Processing question {i}/{total_questions}: {question_id}")
        
        response = ask_question(
            question_data['question'],
            model,
            api_key
        )
        
        output_entry = {
            "question_id": question_id,
            "category": question_data['category'],
            "question": question_data['question'],
            "model": model,
            "timestamp": datetime.datetime.now().isoformat(),
            "response": response
        }
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
        
        # Mark this question as processed
        processed_ids.add(question_id)
        
        time.sleep(1)
    
    print(f"Responses saved to: {output_file}")

def main():
    # Check for API key in environment based on model name
    parser = argparse.ArgumentParser(description='Ask questions to AI models via OpenRouter or Fireworks API')
    parser.add_argument('model', help='Model identifier (e.g., deepseek/deepseek-r1 or accounts/fireworks/models/deepseek-v3)')
    parser.add_argument('questions_files', nargs='+', help='One or more JSONL files containing questions')
    parser.add_argument('--force-restart', action='store_true', help='Ignore existing output file and start from beginning')
    
    args = parser.parse_args()
    
    # Determine which API key to use based on model name
    env_var = 'FIREWORKS_API_KEY' if args.model.startswith('accounts/fireworks') else 'OPENROUTER_API_KEY'
    api_key = os.getenv(env_var)
    
    if not api_key:
        raise ValueError(f"{env_var} environment variable must be set")
    
    for questions_file in args.questions_files:
        try:
            process_questions(
                questions_file,
                args.model,
                api_key,
                args.force_restart
            )
        except Exception as e:
            print(f"Error processing {questions_file}: {str(e)}")

if __name__ == "__main__":
    main()
