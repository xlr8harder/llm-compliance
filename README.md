for model in `cat models.txt` ; do echo $model;  python ask.py $model questions/*.jsonl ; done 
python judge_compliance.py openai/gpt-4o-2024-11-20 responses/*.jsonl
cat analysis/compliance_china_criticism_deepseek_deepseek-chat.jsonl | jq 'select(.compliance == "DENIAL")'
cat analysis/compliance_china_criticism_deepseek_deepseek-chat.jsonl | jq 'select(.compliance == "EVASIVE")'
