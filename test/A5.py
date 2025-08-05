import json
import re

from langchain_core.tracers import ConsoleCallbackHandler

from retriever.rag_evaluate import questions
from retriever.rag_retriever_enhanced import build_rag_chain

if __name__ == '__main__':
    rag_chain = build_rag_chain()
    QAS = []

    for question in questions:
        result = rag_chain.invoke({"query": question["question"]}, config={"callbacks": [ConsoleCallbackHandler()]})
        question["response"] = re.sub(r'<think>.*?</think>', '', result["result"], flags=re.DOTALL)
        question["contexts"] = result["metadata"]["contexts"]
        question["result"] = result
        QAS.append({
            'id': question["id"],
            'reference': question["reference"],
            'question': question["question"],
            'answer': question["answer"],
            'response': question["response"],
            'contexts': question["contexts"],
            'result': question["result"],
        })
    with open('./result/A5.jsonl', 'w', encoding='utf-8') as f:
        for QA in QAS:
            f.write(json.dumps(QA, ensure_ascii=False) + '\n')
