import json
import re

from langchain_core.tracers import ConsoleCallbackHandler

from llm.llm import load_llm
from retriever.rag_evaluate import questions, auto_evaluate_rag
from retriever.rag_retriever_enhanced import build_rag_chain

if __name__ == '__main__':
    rag_chain = build_rag_chain()
    QAS = []
    llm = load_llm()
    for question in questions:
        result= rag_chain.invoke({"query": question["question"]}, config={"callbacks": [ConsoleCallbackHandler()]})
        print(result)
        cleaned = re.sub(r'<think>.*?</think>', '', result["result"], flags=re.DOTALL)
        print(cleaned)
        evaluation = auto_evaluate_rag(question["question"], question["answer"], cleaned, question["reference"],result["metadata"]["contexts"], llm)
        print(evaluation)
        QAS.append({
            'id': question["id"],
            '类型': result["metadata"]["type"],
            '引用': result["metadata"]["sources"],
            '上下文':result["metadata"]['contexts'],
            '问题': question["question"],
            '标准答案': question["answer"],
            '大模型回答': re.sub(r'<think>.*?</think>', '', result['result'], flags=re.DOTALL),
            '评估': evaluation
        })
    with open('./result/A5.jsonl', 'w', encoding='utf-8') as f:
        for QA in QAS:
            f.write(json.dumps(QA, ensure_ascii=False) + '\n')

