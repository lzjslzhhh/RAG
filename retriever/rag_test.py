import pytest
from unittest.mock import MagicMock

from langchain_core.prompts import PromptTemplate

from vectorstore.qdrant_store import load_qdrant_vectorstore


def test_prompt_template():
    template = "参考：{context}\n请回答：{question}"
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # 测试变量填充
    filled = prompt.format(context="测试内容", question="测试问题")
    assert "参考：测试内容" in filled
    assert "请回答：测试问题" in filled

    # 测试缺少变量时的行为
    with pytest.raises(KeyError):
        prompt.format(context="缺少问题变量")

def test_retriever():
    # 模拟向量数据库
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock(
        return_value=[
            {"page_content": "文档1内容", "metadata": {"source": "doc1"}},
            {"page_content": "文档2内容", "metadata": {"source": "doc2"}}
        ]
    )

    # 构建测试链
    test_chain = build_rag_chain(
        template="{context}\n问题：{question}",
        vectorstore=mock_vectorstore
    )

    # 执行测试查询
    result = test_chain.invoke({"query": "测试问题"})

    # 验证返回结构
    assert "result" in result
    assert len(result["source_documents"]) == 2
    assert "doc1" in str(result["source_documents"])


def build_rag_chain(template, vectorstore=None):
    vectorstore = vectorstore or load_qdrant_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 3,})

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
