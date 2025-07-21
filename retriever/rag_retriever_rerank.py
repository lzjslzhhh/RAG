from langchain.prompts import PromptTemplate
from langsmith import traceable
from llm.llm import load_llm
from vectorstore.qdrant_store import load_qdrant_vectorstore
from langchain_core.documents import Document
import heapq
import os


model_path = ''

def rerank_documents(docs, question, llm, top_k=3):
    """使用 LLM 对初步检索的文档打分，选出 top_k 条最相关的。"""

    reranked = []
    for doc in docs:
        content = doc.page_content.strip().replace("\n", " ")
        prompt = f"""
你是一个评估专家，请判断下列文档与提问之间的语义相关性，请返回一个0~1之间的小数表示其相关性程度（越高越相关，不要输出其他内容）。

文档内容：
\"\"\"{content}\"\"\"

提问：
\"\"\"{question}\"\"\"

请仅输出一个小数：
"""
        score_text = llm.invoke(prompt,enable_thinking=False).strip()
        try:
            score = float(score_text)
        except:
            score = 0.0
        reranked.append((score, doc))

    # 选出分数最高的 top_k 个文档
    top_docs = [doc for _, doc in heapq.nlargest(top_k, reranked, key=lambda x: x[0])]
    return top_docs


def build_rag_chain():
    # 加载向量库与模型
    vectorstore = load_qdrant_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 10})
    llm = load_llm()

    prompt_template = PromptTemplate(
        input_variables=['context', 'question'],
        template="""
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

请开始逐步推理并给出答案：
{question}

"""
    )

    def rag_pipeline(question: str):
        # Step 1: 向量召回前10
        initial_docs = retriever.get_relevant_documents(question)

        # Step 2: LLM rerank，取top-3
        top_docs = rerank_documents(initial_docs, question, llm, top_k=3)

        # Step 3: 拼接上下文
        context = "\n\n".join([doc.page_content for doc in top_docs])
        prompt = prompt_template.format(context=context, question=question)

        # Step 4: 推理生成答案
        answer = llm.predict(prompt)

        return {
            "answer": answer.strip(),
            "source_documents": top_docs
        }

    return rag_pipeline


# rag_chain = build_rag_chain()
#
#
# @traceable(name="GridRAGChain")
# def run_rag(question: str):
#     return rag_chain(question)
