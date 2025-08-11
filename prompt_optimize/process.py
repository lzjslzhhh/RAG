import json

question_type='判断题'
with open(fr'/tmp/pycharm_project_581/EleQA-master/issues/{question_type}.json','r',encoding='utf-8') as f:
    questions = json.load(f)
    questions = [{
        'question': question['question'],
        'answer': question['answer']}
        for question in questions
    ]
with open(fr'/tmp/pycharm_project_581/EleQA-master/{question_type}/train.jsonl','w',encoding='utf-8') as out_f:
    for question in questions[:4000]:
        out_f.write(json.dumps(question, ensure_ascii=False) + '\n')
with open(fr'/tmp/pycharm_project_581/EleQA-master/{question_type}/test.jsonl','w',encoding='utf-8') as out_f:
    for question in questions[4000:6150]:
        out_f.write(json.dumps(question, ensure_ascii=False) + '\n')