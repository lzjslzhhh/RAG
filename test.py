from llm.llm import load_llm

llm = load_llm()

print(llm.invoke("你是什么模型?",enable_thinking=False))