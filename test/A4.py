import json

from langchain_core.tracers import ConsoleCallbackHandler

from retriever.rag_retriever import build_rag_chain


# '光伏发电系统接入配电网时如何进行防孤岛保护检测?'光伏发电系统接入配电网检测规程.pdf11页
# '电化学储能电站接入电网的额定能量如何进行测试?'电化学储能电站接入电网测试规程.pdf13页
# '风力发电机在电网中的谐波电压适应性如何测试?'风力发电机组%20电网适应性测试规程.pdf第12页

if __name__ == '__main__':
    rag_chain = build_rag_chain(template="""
请你扮演一位具有深厚电力系统背景的智能助手，针对电网相关的技术规程、检测标准、控制规范等文档内容，进行**严谨、分步骤**的推理和问答。请务必严格依赖提供的上下文，不得编造内容。

背景材料如下：
{context}

---

请按照以下步骤进行推理并回答问题：

1. **理解问题语义**：明确提问中涉及的技术概念、规程条款或控制流程；
2. **定位上下文依据**：在背景材料中查找相关条款、参数范围或操作规则；
3. **分条分析内容**：逐条解释与问题相关的规范内容，若有操作步骤或技术判断，请清晰列出；
4. **综合推导答案**：在推理基础上，得出符合规程的明确结论；
5. **输出最终答案**：用简洁、规范的术语回答问题。
6. **请用纯文本格式回答**: 不要包含Markdown、代码块等特殊格式。
7. **可能存在OCR识别错误**（如："整定值"可能被误识别为"设定值"）：请按你的理解改正

请开始逐步推理并给出答案：
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
        print("回答：", result["result"])
        # print("检索来源：")
        # for doc in result["source_documents"]:
        #     print(" -", doc.metadata.get("source", "无"))
        QAS.append({
            '问题': query,
            '回答': result["result"],
            # '检索来源':result['source_documents']
        })
    with open('./result/A4_2.jsonl', 'w',encoding='utf-8') as f:
        f.write(json.dumps(QAS, ensure_ascii=False))
