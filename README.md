# LLM compliance testing
This is the code and data I used to check various LLMs for compliance in requests to compose political speech critical of various governments.

Final result:
[compliance graph](report/government_criticism_analysis.png)

Reproduction:
```bash
export OPENROUTER_API_KEY=...
for model in `cat models.txt` ; do echo $model;  python ask.py $model questions/*.jsonl & done 
python judge_compliance.py openai/gpt-4o-2024-11-20 responses/*.jsonl

cat analysis/compliance_china_criticism_deepseek_deepseek-chat.jsonl | jq 'select(.compliance == "DENIAL")'
cat analysis/compliance_china_criticism_deepseek_deepseek-chat.jsonl | jq 'select(.compliance == "EVASIVE")'

python report_compliance_categories.py analysis/*.jsonl
# final result in reports/government_criticism_analysis.png 
```

