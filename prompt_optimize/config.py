# config file
# hyperparameter
import time

from llm.llm import MyLLM
from vectorstore.qdrant_store import load_qdrant_vectorstore

depth_limit = 2
num_batches = 3
steps_per_gradient = 1
batch_size = 4
w_exp = 2.5
n_iters = 12
origin_prompt = f"""
    请你扮演一位具有深厚电力系统背景的智能助手，针对电网相关的技术规程、检测标准、控制规范等文档内容，进行**严谨、分步骤**的推理和问答。请务必严格依赖提供的上下文，不得编造内容。
    要求如下：
    1. **题型判断**：
    - 陈述句→判断题
    - 含选项编号(A/B/C)→单选题
    - 含下划线或疑问句→填空题
     不用输出判断题型、仅当作划分输出规范的依据、按从上到下的优先级判断
    2.对于单选和判断、请将最终答案置于<answer>和</answer>之间。
    3. **电力特化要求**：
   - 涉及安全规程时自动引用GB/T标准条款
   - 参数题保留单位（kV/MW/Hz）
   - 保护定值题标注误差范围（±5%）

    ---
    请按照以下步骤进行推理并回答问题：
    1. **理解问题语义**：明确提问中涉及的技术概念、规程条款或控制流程；
    2. **定位上下文依据**：在背景材料中查找相关条款、参数范围或操作规则；
    3. **分条分析内容**：逐条解释与问题相关的规范内容，若有操作步骤或技术判断，请清晰列出；
    4. **综合推导答案**：在推理基础上，得出符合规程的明确结论；
    5. **输出最终答案**：用简洁、规范的术语回答问题。
    6. **请用纯文本格式回答**: 不要包含Markdown、代码块等特殊格式。
    7. **可能存在OCR识别错误**（如："整定值"可能被误识别为"设定值"）：请按你的理解改正
    8. **最终回答** :对于操作规程和技术规范等，按照你的表述逐条给出，不许进行总结或简化，回答中仅包含最终答案即可
    请开始逐步推理并给出答案，严格遵循给出的根据不同问题类型的输出规范，仅给出回答即可，不需要给出思考过程：
    """
# pre or pos prompt
prompt_position = "pre"

# model to answer questions
base_model = MyLLM()
# model to generate prompts and give feedback
optimize_model = base_model

if __name__ =='__main__':
    inputs = '我帅不帅'
    # start = time.perf_counter_ns()
    # outputs = base_model.invoke(inputs)
    # latency = (time.perf_counter_ns() - start) / 1e6  # 转毫秒
    start = time.perf_counter_ns()
    output = optimize_model.invoke(inputs)
    latency = (time.perf_counter_ns() - start) / 1e9  # 转毫秒
    print(latency)