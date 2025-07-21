from llm.llm import load_llm
from retriever.rag_retriever import build_rag_chain, run_rag

if __name__ == '__main__':
    # rag_chain = build_rag_chain()
    # llm = load_llm()
    while True:
        query = input('问点什么吧')
        if query.strip().lower() in ["exit", "quit"]:
            break
        result = run_rag(query)
        # result = rag_chain({"query": query})
        # response = llm.invoke(query)
        # print(response)
        print("回答：", result["result"])
        print("检索来源：")
        for doc in result["source_documents"]:
            print(" -", doc.metadata.get("source", "无"))
