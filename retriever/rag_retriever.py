from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langsmith import traceable
from llm.llm import load_llm
from vectorstore.qdrant_store import load_qdrant_vectorstore
from langchain_core.tracers import LangChainTracer
import os


def build_rag_chain(template):
    vectorstore = load_qdrant_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    test_docs = retriever.invoke("光伏发电防孤岛保护")
    print(f"检索到文档数: {len(test_docs)}")
    for doc in test_docs:
        print(doc.page_content[:100] + "...")
        print("元数据:", doc.metadata)
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=template,
    )

    llm = load_llm()

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            'prompt': prompt
        }
    )
    return rag_chain

# rag_chain = build_rag_chain()
#
# @traceable(name="GridRAGChain")
# def run_rag(question):
#     rag_train = build_rag_chain()
#     return rag_train.invoke({'question': question})
