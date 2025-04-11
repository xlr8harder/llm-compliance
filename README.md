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
find analysis -name "*.jsonl" -not -name "compliance_zh*.jsonl" -and -not -name "compliance_fi*.jsonl" | xargs python report.py -o report/government_criticism_analysis.png --sort-by compliance

# for china questions only
python report.py --sort-by compliance  -o report/multilingual_china_criticism.png analysis/*china*.jsonl


# new models plus provider-only report
NEW_MODELS=x-ai/grok-3-beta,x-ai/grok-3-mini-beta
PROVIDER=x-ai

find analysis -name "*.jsonl" -not -name "compliance_zh*.jsonl" -and -not -name "compliance_fi*.jsonl" |  xargs python report.py -o report/government_criticism_analysis.png --highlight-models $NEW_MODELS --sort-by compliance
python report.py --sort-by compliance -o report/multilingual_china_criticism.png --highlight-models $NEW_MODELS analysis/*china*.jsonl
find analysis -name "*.jsonl" -not -name "compliance_zh*.jsonl" -and -not -name "compliance_fi*.jsonl" | grep $PROVIDER | xargs python report.py -o report/${PROVIDER}_government_criticism_analysis.png --highlight-models $NEW_MODELS --sort-by compliance
python report.py --sort-by compliance -o report/${PROVIDER}_multilingual_china_criticism.png --highlight-models $NEW_MODELS analysis/*china*${PROVIDER}*.jsonl


```
