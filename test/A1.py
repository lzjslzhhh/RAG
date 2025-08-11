import json
import os

from llm.llm import load_llm
from retriever.rag_embedding import COLLECTION_NAME
from eval_config import question_type,question_cots
from vectorstore.qdrant_store import load_qdrant_vectorstore

if __name__ == '__main__':
    llm = load_llm(enable_thinking=False)
    QAS = []
    vectorstore, client = load_qdrant_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    dir_path = '/tmp/pycharm_project_581/EleQA-master/issues'
    file_path = os.path.join(dir_path, f'{question_type}.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)[:question_cots]
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

            prompt = """请用纯文本格式回答，不要包含Markdown、代码块等特殊格式：
                    1. **题型判断**：
                   - 陈述句→判断题
                   - 含选项编号(A/B/C)→单选题
                   - 含下划线或疑问句→填空题
                   - 其他→正常回答即可
                    不用输出判断题型、仅当作划分输出规范的依据、按从上到下的优先级判断
                    2.请将最终答案置于<answer>和</answer>之间。
                    {}
                    """.format(question["question"])
            llm_answer = llm.invoke(prompt,enable_thinking=False,presence_penalty=1.2)
            print(llm_answer)
            QAS.append({
                'type': question["type"],
                'reference': question["reference"],
                'referenceId': question["referenceId"],
                'question': question["question"],
                'answer': question["answer"],
                'docId': question["docId"],
                'response': llm_answer,
                'contexts': [doc.page_content for doc in full_docs],
            })
    with open(f'./result/A1_{question_type}.jsonl', 'w', encoding='utf-8') as f:
        for QA in QAS:
            f.write(json.dumps(QA, ensure_ascii=False) + '\n')
