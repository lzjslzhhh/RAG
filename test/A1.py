import json

from llm.llm import load_llm

# '光伏发电系统接入配电网时如何进行防孤岛保护检测?'光伏发电系统接入配电网检测规程.pdf11页
# '电化学储能电站接入电网的额定能量如何进行测试?'电化学储能电站接入电网测试规程.pdf13页
# '风力发电机在电网中的谐波电压适应性如何测试?'风力发电机组%20电网适应性测试规程.pdf第12页
if __name__ == '__main__':
    llm = load_llm()
    queries = [
        '光伏发电系统接入配电网时如何进行防孤岛保护检测?',
        '电化学储能电站接入电网的额定能量如何进行测试?',
        '风力发电机在电网中的谐波电压适应性如何测试?'
    ]
    QAS=[]
    for query in queries:
        prompt = """请用纯文本格式回答，不要包含Markdown、代码块等特殊格式：
        {}
        """.format(query)
        response = llm.invoke(prompt, enable_thinking=False)
        print(response)
        QAS.append({
                '问题': query,
                '回答': response
            })
    with open('./result/A1.jsonl', 'w', encoding='utf-8') as f:
        f.write(json.dumps(QAS, ensure_ascii=False))
