# LLM compliance testing
This is the code and data I used to check various LLMs for compliance in requests to compose political speech critical of various governments.

Results - English
![compliance graph](report/government_criticism_analysis.png)

Results - China Criticism, English and Chinese
![china compliance graph](report/multilingual_china_criticism.png)

Reproduction:
```bash
export OPENROUTER_API_KEY=...
for model in `cat models.txt` ; do echo $model;  python ask.py $model questions/*.jsonl & done 
python judge_compliance.py openai/gpt-4o-2024-11-20 responses/*.jsonl

cat analysis/compliance_china_criticism_deepseek_deepseek-chat.jsonl | jq 'select(.compliance == "DENIAL")'
cat analysis/compliance_china_criticism_deepseek_deepseek-chat.jsonl | jq 'select(.compliance == "EVASIVE")'
cat analysis/compliance_china_criticism_deepseek_deepseek-chat.jsonl | jq 'select(.compliance == "ERROR")'
cat analysis/compliance_china_criticism_deepseek_deepseek-chat.jsonl | jq 'select(.compliance == "INVALID")'

# report on english language questions
find analysis -name "*.jsonl" -not -name "compliance_zh*.jsonl" -and -not -name "compliance_fi*.jsonl" | xargs python report.py -o report/government_criticism_analysis.png

# for china questions only
python report.py -o report/multilingual_china_criticism.png analysis/*china*.jsonl
```
