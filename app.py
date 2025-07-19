from llm.llm import load_llm
from retriever.rag_retriever import build_rag_chain

if __name__ == '__main__':
    # rag_chain = build_rag_chain()
    llm = load_llm()
    while True:
        query = '电力监控系统的安全保护措施'
        # if query.strip().lower() in ["exit", "quit"]:c
        #     break
        # result = rag_chain({"query": query})
        response = llm.invoke(query)
        print(response)
        # print("回答：", result["result"])
        # print("检索来源：")
        # for doc in result["source_documents"]:
        #     print(" -", doc.metadata.get("source", "无"))
