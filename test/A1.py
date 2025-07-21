import json
from llm.llm import load_llm


if __name__ == '__main__':
    while True:
        llm = load_llm()
        query = input('问点什么吧')
        if query.strip().lower() in ["exit", "quit"]:
            break
        response = llm.invoke(query,enable_thinking=False)
        print(response)
        with open('./result/A1.jsonl', 'w',encoding='utf-8') as f:
            f.write(json.dumps({
                '问题': query,
                '回答': response
            }, ensure_ascii=False))

