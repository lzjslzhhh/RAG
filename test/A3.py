import json

from retriever.rag_retriever import build_rag_chain

if __name__ == '__main__':
    rag_chain = build_rag_chain()
    while True:
        query = input('问点什么吧')
        if query.strip().lower() in ["exit", "quit"]:
            break
        result = rag_chain({"query": query})
        print("回答：", result["result"])
        print("检索来源：")
        for doc in result["source_documents"]:
            print(" -", doc.metadata.get("source", "无"))
        with open('./result/A3.jsonl', 'w',encoding='utf-8') as f:
            f.write(json.dumps({
                '问题': query,
                '回答': result["result"],
                '检索来源':result['source_documents']
            }, ensure_ascii=False))
