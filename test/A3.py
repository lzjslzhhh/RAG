import json

from langchain_core.tracers import ConsoleCallbackHandler

from retriever.rag_retriever import build_rag_chain


# '光伏发电系统接入配电网时如何进行防孤岛保护检测?'光伏发电系统接入配电网检测规程.pdf11页
# '电化学储能电站接入电网的额定能量如何进行测试?'电化学储能电站接入电网测试规程.pdf13页
# '风力发电机在电网中的谐波电压适应性如何测试?'风力发电机组%20电网适应性测试规程.pdf第12页

if __name__ == '__main__':
    rag_chain = build_rag_chain(template="""
        请用纯文本格式回答，不要包含Markdown、代码块等特殊格式,可能存在ocr识别错误（如："整定值"可能被误识别为"设定值"）,请按照你的理解纠正：
        {context}

        请根据以上信息回答问题：
        
        {question}
        """)

    queries = [
        '光伏发电系统接入配电网时如何进行防孤岛保护检测?',
        '电化学储能电站接入电网的额定能量如何进行测试?',
        '风力发电机在电网中的谐波电压适应性如何测试?'
    ]
    QAS=[]
    for query in queries:
        result = rag_chain.invoke({"query": query},config={"callbacks": [ConsoleCallbackHandler()]})
        print(result)
        print("回答：", result["result"])
        # print("检索来源：")
        # for doc in result["source_documents"]:
        #     print(" -", doc.metadata.get("source", "无"))
        QAS.append({
            '问题': query,
            '回答': result["result"],
            # '检索来源':result['source_documents']
        })

    with open('./result/A3.jsonl', 'w',encoding='utf-8') as f:
        f.write(json.dumps(QAS, ensure_ascii=False))
