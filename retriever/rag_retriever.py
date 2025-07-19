from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from llm.llm import load_llm, RemoteLLM
from vectorstore.qdrant_store import load_qdrant_vectorstore


def build_rag_chain():
    vectorstore = load_qdrant_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 3})

    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template="""
        请根据以下背景知识回答问题，逐步推理，输出推理过程和最终答案。
        {context}

        请根据以上信息回答问题：
        {question}
        """,
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
