import json

with open(r'D:\learning\RAG\RAG\EleQA-master\issues\单选题.json','r',encoding='utf-8') as f:
    questions = json.load(f)
    questions = [{
        'question': question['question'],
        'answer': question['answer']}
        for question in questions
    ]
with open(r'D:\learning\RAG\RAG\EleQA-master\ele_qa\train.jsonl','w',encoding='utf-8') as out_f:
    for question in questions[:2150]:
        out_f.write(json.dumps(question, ensure_ascii=False) + '\n')
with open(r'D:\learning\RAG\RAG\EleQA-master\ele_qa\test.jsonl','w',encoding='utf-8') as out_f:
    for question in questions[2150:2650]:
        out_f.write(json.dumps(question, ensure_ascii=False) + '\n')