import json
from llm.llm import load_llm
from retriever.rag_evaluate import auto_evaluate_llm, questions

if __name__ == '__main__':

    llm = load_llm()
    QAS=[]
    for question in questions:
        prompt = """请用纯文本格式回答，不要包含Markdown、代码块等特殊格式：
                {}
                """.format(question["question"])
        llm_answer = llm.invoke(prompt, enable_thinking=False)
        print(llm_answer)
        evaluation = auto_evaluate_llm(question["question"],question["answer"],llm_answer,question["reference"],llm)
        print(evaluation)
        QAS.append({
            'id': question["id"],
            'type': question["type"],
            '问题': question["question"],
            '标准答案': question["answer"],
            '大模型回答': llm_answer,
            '评估':evaluation
        })
    with open('./result/A1.jsonl', 'w',encoding='utf-8') as f:
        for QA in QAS:
            f.write(json.dumps(QA, ensure_ascii=False)+'\n')

