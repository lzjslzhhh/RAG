import json
import json
import os
import re

from langchain_core.tracers import ConsoleCallbackHandler

from retriever.rag_evaluate import questions
from retriever.rag_retriever import build_rag_chain

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 强制同步执行内核

# 启用设备端断言
os.environ['TORCH_USE_CUDA_DSA'] = '1'
if __name__ == '__main__':
    # queries = [
    #     '光伏发电系统接入配电网时如何进行防孤岛保护检测?',
    #     '电化学储能电站接入电网的额定能量如何进行测试?',
    #     '风力发电机在电网中的谐波电压适应性如何测试?'
    # ]
    rag_chain = build_rag_chain(True)
    QAS = []
    for question in questions:
        result = rag_chain.invoke({"query": question["question"]}, config={"callbacks": [ConsoleCallbackHandler()]})
        # print(re.sub(r'<think>.*?</think>', '', result["result"], flags=re.DOTALL))
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
    with open('./result/A3.jsonl', 'w', encoding='utf-8') as f:
        for QA in QAS:
            f.write(json.dumps(QA, ensure_ascii=False) + '\n')
