import json
import os
import re

from langchain_core.tracers import ConsoleCallbackHandler
from retriever.rag_retriever_enhanced import build_rag_chain
from eval_config import question_type, question_cots

if __name__ == '__main__':
    rag_chain = build_rag_chain()
    QAS = []
    dir_path = '/tmp/pycharm_project_581/EleQA-master/issues'
    file_path = os.path.join(dir_path, f'{question_type}.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)[:question_cots]
        for question in questions:
            result = rag_chain.invoke({"query": question["question"]}, config={"callbacks": [ConsoleCallbackHandler()]})
            question["response"] = re.sub(r'<think>.*?</think>', '', result["result"], flags=re.DOTALL)
            question["contexts"] = result["metadata"]["contexts"]
            question["result"] = result
            QAS.append({
                'id': question["id"],
                'reference': question["reference"],
                'referenceId': question["referenceId"],
                'question': question["question"],
                'answer': question["answer"],
                'docId': question["docId"],
                'response': question["response"],
                'contexts': question["contexts"],
                'result': question["result"],
            })
    with open(f'./result/A5_{question_type}.jsonl', 'w', encoding='utf-8') as f:
        for QA in QAS:
            f.write(json.dumps(QA, ensure_ascii=False) + '\n')
