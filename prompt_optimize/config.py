# config file
# hyperparameter
import time

from llm.llm import MyLLM

depth_limit = 4
origin_prompt = "请用你在电网领域的知识回答问题"
num_batches = 3
steps_per_gradient = 1
batch_size = 5
w_exp = 2.5
n_iters = 12
# pre or pos prompt
prompt_position = "pre"

# model to answer questions
base_model = MyLLM(enable_thinking=False)
# model to generate prompts and give feedback
optimize_model = base_model


if __name__ =='__main__':
    inputs = '你是什么模型'
    # start = time.perf_counter_ns()
    # outputs = base_model.invoke(inputs)
    # latency = (time.perf_counter_ns() - start) / 1e6  # 转毫秒
    start = time.perf_counter_ns()
    output = optimize_model.invoke(inputs)
    latency = (time.perf_counter_ns() - start) / 1e6  # 转毫秒
    print(latency)