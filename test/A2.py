import json
import re

from llm.llm import load_llm
from retriever.rag_embedding import COLLECTION_NAME
from retriever.rag_evaluate import questions
from vectorstore.qdrant_store import load_qdrant_vectorstore

if __name__ == '__main__':

    llm = load_llm()
    QAS = []
    vectorstore, client = load_qdrant_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    for question in questions:
        docs = retriever.invoke(question["question"])
        full_docs = []
        for doc in docs:
            point_id = doc.metadata["_id"]
            result = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[point_id],
                with_payload=True
            )[0]
            doc.metadata.update(result.payload)
            full_docs.append(doc)
        prompt = """请用纯文本格式回答，不要包含Markdown、代码块等特殊格式，仅给出回答即可，不需要给出思考过程：
                {}
                """.format(question["question"])
        llm_answer = llm.invoke(prompt, enable_thinking=True)
        print(re.sub(r'<think>.*?</think>', '', llm_answer, flags=re.DOTALL))
        QAS.append({
            'id': question["id"],
            'reference': question["reference"],
            'question': question["question"],
            'answer': question["answer"],
            'response': re.sub(r'<think>.*?</think>', '', llm_answer, flags=re.DOTALL),
            'contexts': [doc.page_content for doc in full_docs],
        })
    with open('./result/A2.jsonl', 'w', encoding='utf-8') as f:
        for QA in QAS:
            f.write(json.dumps(QA, ensure_ascii=False) + '\n')
