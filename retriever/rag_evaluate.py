import pandas as pd
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModel
from transformers import AutoTokenizer


def build_eval_prompt(question, standard_answer, llm_answer, standard_text):
    return f"""
【任务说明】
你是电网行业的技术审查专家，具备国家标准和工程实践经验。请你对一个大语言模型生成的回答，从**技术、合规、适用性与表达规范**四个方面进行全面评估，并输出总评分（满分100）和模型评分的**可信度（Confidence）**。

---

【输入信息】
● 真实问题：
{question}

● 标准依据内容：
{standard_text}

● 标准答案（关键词参考）：
{standard_answer}

● 模型生成回答（被评估对象）：
{llm_answer}

---

【评估维度说明】

1. 📐 技术准确性（40分）
   - 是否完整、正确引用标准编号与条款号（如：GB/T 36548 第7.3条）；
   - 是否复述标准原文时保持原意，尤其是数值类指标（如±5%）是否精度正确；
   - 是否合理引用多个标准并指出优先级或适用条件；
   - 若使用动态Prompt生成的回答，是否结构清晰、分条陈述更合理。

2. ⚠️ 安全合规性（30分）
   - 是否体现关键安全要求，如响应时限、谐波限值、电压下限等；
   - 是否考虑异常或边界场景（如“电网恢复后是否立即闭合”等）；
   - 是否遗漏任何对人员、系统造成安全隐患的关键要求。

3. 🛠 工程适用性（20分）
   - 回答能否直接应用于测试现场或技术指导；
   - 操作类问题是否有明确步骤/先后逻辑，参数类问题是否标明判据；
   - 是否指出了测试条件、判断依据等实际可执行要素。

4. ✏️ 表达规范性（10分）
   - 是否使用规范术语（如“低电压穿越”“频率偏移”）；
   - 是否标注单位（如 MW/kV/Hz）；
   - 回答是否条理清晰，避免含糊表达，如“应满足要求”或“数值合适”等。

---

【评分标准】

| 评分等级 | 技术准确性 | 安全合规性 | 工程适用性 | 表达规范 | 综合得分 | 置信度 |
|----------|--------------|---------------|------------------|--------------|----------------|-------------|
| ★★★★★     | ≥36           | 满分30         | ≥16              | ≥9           | 90~100分       | ≥95%        |
| ★★★★☆     | 32~35         | ≥28            | ≥14              | ≥8           | 80~89分        | 90~94%      |
| ★★★☆☆     | 28~31         | ≥25            | ≥12              | ≥7           | 70~79分        | 80~89%      |
| ★★☆☆☆     | 24~27         | ≥20            | ≥10              | ≥6           | 60~69分        | 70~79%      |
| ★☆☆☆☆     | <24           | <20            | <10              | <6           | <60分          | <70%        |

---

【输出格式（必须为纯文本，不使用 Markdown）】

技术准确性：
- 得分（满分40）：__ /40
- 评语（最多3条）：
  1. ...
  2. ...
  3. ...

安全合规性：
- 得分（满分30）：__ /30
- 是否满足关键安全要求（是/否）：__
- 问题说明：

工程适用性：
- 得分（满分20）：__ /20
- 可执行性评价：__

表达规范性：
- 得分（满分10）：__ /10
- 表述建议：

最终综合评分：
- 综合得分（满分100）：__
- 置信度（%）：__
- 评定星级：__（如 ★★★★☆）

改进建议（如有）：
1. ...
2. ...
"""


def sim_by_gte(standard_answer, llm_answer):
    input_texts = [standard_answer, llm_answer]
    model_name_or_path = r"/tmp/pycharm_project_581/gte-multilingual-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)

    dimension = 768  # The output dimension of the output embedding, should be in [128, 768]
    embeddings = outputs.last_hidden_state[:, 0][:dimension]
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return (embeddings[:1] @ embeddings[1:].T) * 100


def prepare_dataset(question, standard_answer, llm_answer, standard_text):
    data = pd.DataFrame([{
        'question': question,
        'answer': llm_answer,
        'contexts': standard_text,
        'ground_truth': standard_answer,
    }])
    return Dataset.from_pandas(data)


def auto_evaluate_llm(question, standard_answer, llm_answer, standard_text, llm):
    llm_eval = llm.invoke(build_eval_prompt(question, standard_answer, llm_answer, standard_text), enable_thinking=True)
    # P,R,F1 = score([standard_answer],[llm_answer],lang='zh')
    sim_score = sim_by_gte(standard_answer, llm_answer)
    # dataset = prepare_dataset(question, standard_answer, llm_answer, standard_text)
    # metrics = [
    #     answer_relevancy,
    #     faithfulness,
    #     context_recall,
    #     context_precision,
    # ]
    # ragas_eval = evaluate(dataset,metrics=metrics)
    import re
    return {
        # "overall": {
        #     "ragas_score": ragas_eval["ragas_score"],
        #     "metric_count": len(metrics)
        # },
        # "details": ragas_eval.to_pandas().to_dict(orient="records")[0],
        "sim_score": sim_score.item(),
        "llm_eval": re.sub(r'<think>.*?</think>', '', llm_eval, flags=re.DOTALL),
    }


def auto_evaluate_rag(question, standard_answer, llm_answer, standard_text, retrieved_context, llm):
    llm_eval = llm.invoke(build_eval_prompt(question, standard_answer, llm_answer, standard_text), enable_thinking=True)
    # P,R,F1 = score([standard_answer],[llm_answer],lang='zh')
    sim_score = sim_by_gte(standard_answer, llm_answer)
    # dataset = prepare_dataset(question, standard_answer, llm_answer, retrieved_context)
    # metrics = [
    #     answer_relevancy,
    #     faithfulness,
    #     context_recall,
    #     context_precision,
    # ]
    # embedding_model = HuggingFaceEmbeddings(
    #     model_name=MODEL_PATH,
    #     model_kwargs={"device": "cuda","trust_remote_code": True},
    #     encode_kwargs={"normalize_embeddings": True},
    # )
    # ragas_eval = evaluate(dataset,metrics=metrics,embeddings=embedding_model,llm=llm)
    import re
    return {
        # "ragas_eval":ragas_eval,
        "sim_score": sim_score.item(),
        "llm_eval": re.sub(r'<think>.*?</think>', '', llm_eval, flags=re.DOTALL),
    }


# 微电网接入配电网测试规范.pdf
questions = [
    {
        "id": "Q1",
        "type": "标准条款",
        "question": "微电网进行防孤岛保护功能测试时，响应时间应满足的标准要求？",
        "answer": "响应时间应≤2秒",
        "reference": """
                （GB/T 34129-2017 第7.6.2条）
                a) 将被测微电网与配电网相连,所有参数调至被测微电网正常工作条件;
                b) 调节微电网与配电网之间联络线交换功率,直至联络线交换功率接近零为止;
                c) 拉开被测微电网并网点开关的上一级开关,并记录开关断开时刻t1;
                d) 记录微电网并网点开关断开时刻 t2;
                e) 计算被测微电网防孤岛保护响应时间(t2 — t1) ;
                f) 试验重复 3 次,任一次响应时间超过 2 s,则测试不通过。
            """
    },
    # {
    #     "id": "Q2",
    #     "type": "计算验证",
    #     "question": "微电网接入配电网的低电压穿越测试中，三相电压跌落的极值点如何选取？",
    #     "answer": "应在UL1~UL2之间均匀选择7个点（含UL1和UL2），其中UL1为最低跌落点（如20%Un），UL2为正常工作电压下限（如85%Un）",
    #     "reference": """GB/T 34129-2017 7.5.3 d) 电压跌落点在 Ul1 一Ul2之间均匀选择 7 个跌落点且包括 2 个极值点 Ul1 和 Ul2 (Ul1 是低电穿越的电压最低跌落点,Up为正常工作电压的最低值,这两个值按照 GB/T 33589 的规定选取);"""
    # },
    # {
    #     "id": "Q3",
    #     "type": "操作流程",
    #     "question": "储能电站低电压故障穿越的空载测试步骤是什么？",
    #     "answer":
    #         """低电压故障穿越的空载测试按以下步骤进行，
    #             a) 断开被测储能系统与电网模拟装置之间的开关;
    #             b) 设置电网模拟装置的输出电压模拟线路三相对称故障，电压跌落点的选取应满足15.1的要求;
    #             c) 利用数据采集装置采集电压跌落前3 s到电压恢复正常后6Ss之间的储能系统测试点电压和电流 ，
    #             并记录;
    #             d) 重复b) ~c) ;
    #             e) 设置电网模拟装置的输出电压模拟表3中的一种不对称故障类型 ，电压跌落点的选取应满足15.1的要求
    #             f) 利用数据采集装置采集电压跌落前3 s到电压恢复正常后6s之间的储能系统测试点电压和电流 ，
    #             并记录;
    #             g) 重复e) ~f)
    #         """
    #     ,
    #     "reference": "GB/T 36548-2024 15.2.1 测试方法 a)~g)"
    # },
    # {
    #     "id": "Q4",
    #     "type": "技术参数",
    #     "question": "微电网接入配电网的并技术参数网测试若使用模拟电网需满足哪些参数要求？",
    #     "answer":
    #         """a)  可输出的短路电流应不小于被测微电网最大交换电流允许值的 20 倍;
    #         b) 谐波应小于 GB/T 14549 规定的谐波允许值的 50%;
    #         c) 稳态电压变化幅度不得超过额定电压的±1%;
    #         d) 电压偏差应在额定电压的±3%以内;
    #         e) 频率偏差应小于±0.01 Hz;
    #         f) 三相电压不平衡度应小于1% ,相位偏差应小于 1%;
    #         g) 中性点不接地的模拟电网,中性点位移电压应小于相电压额定值的1%;
    #         h) 具有在一个周波内进行±3%额定电压的调节能力;
    #         i) 具有在一个周波内进行±0.1%额定频率的调节能力;
    #         """
    #     ,
    #     "reference": """GB/T 34129-2017 5.2.3 并网测试用模拟电网要求
    #         a)  可输出的短路电流应不小于被测微电网最大交换电流允许值的 20 倍;
    #         b) 谐波应小于 GB/T 14549 规定的谐波允许值的 50%;
    #         c) 稳态电压变化幅度不得超过额定电压的±1%;
    #         d) 电压偏差应在额定电压的±3%以内;
    #         e) 频率偏差应小于±0.01 Hz;
    #         f) 三相电压不平衡度应小于1% ,相位偏差应小于 1%;
    #         g) 中性点不接地的模拟电网,中性点位移电压应小于相电压额定值的1%;
    #         h) 具有在一个周波内进行±3%额定电压的调节能力;
    #         i) 具有在一个周波内进行±0.1%额定频率的调节能力;"""
    # },
    # {
    #     "id": "Q5",
    #     "type": "标准条款",
    #     "question": "光伏发电系统电压适应性测试中，86%额定电压下的保持时间要求是多少？",
    #     "answer": "应保持1分钟",
    #     "reference": """
    #     GB/T30152 7.3 电压适应性
    #     检测应按照如下步骤进行:
    #     a) 在公共连接点标称频率条件下,调节电网模拟装置,使公共连接点电压至86%UN,并保持时间
    #     为1 min,记录光伏发电系统运行时间或脱网跳闸时间 ;
    #     b) 在公共连接点标称频率条件下 ,调节电网模拟装置,使公共连接点电压至 109%UN ,并保持时
    #     间为 1 min,记录光伏发电系统运行时间或脱网跳闸时间 ;
    #     c) 在公共连接点标称频率条件下,调节电网模拟装置,使公共连接点电压至 86 % Un ~ 109 %Un
    #     之间任意值 ,并保持时间为 1 min,记录光伏发电系统运行时间或脱网跳闸时间。
    #     注: UN 为公共连接点标称电压。
    #     """
    # },
    # {
    #     "id": "Q6",
    #     "type": "操作流程",
    #     "question": "光伏发电系统接入配电网时防孤岛保护检测时如何配置RLC负载？",
    #     "answer": "需满足：1)LC无功=系统无功 2)RLC有功=系统有功 3)品质因数1±0.2 4)流过K2的基波电流小于被测光伏发电系统输出电流的5%",
    #     "reference": """
    #     GB/T30152 10.3.2 检测步骤
    #     检测应按照如下步骤进行:
    #     a) 防孤岛能力检测点应选择在光伏发电系统并网点处。
    #     b) 通过功率检测装置测量被测光伏发电系统的有功功率和无功功率输出。
    #     c) 依次投入电感L、电容C 、电阻 R,使得:
    #         1) LC 消耗的无功功率等于被测光伏发电系统发出的无功功率;
    #         2) RLC 消耗的有功功率等于被测光伏发电系统发出的有功功率;
    #         3) RLC 谐振电路的品质因数为 1±0.2;
    #         4) 流过 K2 的基波电流小于被测光伏发电系统输出电流的5%。
    #     d) 断开K2, ,通过数字示波器记录被测光伏发电系统运行情况。
    #     e) 读取数字示波器和功率检测装置数据进行分析,若被测光伏发电系统在 2 s 内停止向交流负
    #     载供电,则不再继续检测 。否则应进行下步检测 。
    #     f) 调节电感L,电容C 使L、C 的无功功率按表 2 的规定每次变化±2%;表 2 中的参数表示的是
    #     图 2 中流经开关 K2 的无功功率流的方向,正号表示功率流从被测光伏发电系统到电网。
    #     g) 每次调节后 ,断开 K2, ,通过数字示波器记录被测光伏发电系统运行情况;若记录的时间呈持续
    #     上升趋势,则应继续以 2%的增量扩大调节范围,直至记录的时间呈下降趋势。
    #     h) 读取数字示波器数据进行分析,输出报表和测量曲线,并判别是否满足 GB/T 29319 的要求，
    #     检测记录见附录 A。
    #     """
    # },
    # {
    #     "id": "Q7",
    #     "type": "技术参数",
    #     "question": "光伏发电系统接入配电网时电压谐波测量时DFT频谱线间隔是多少？",
    #     "answer": "5Hz",
    #     "reference": """
    #     GB/T30152 3.4 时间窗 time window
    #     Tw
    #     测量电流谐波、间谐波所取的时间宽度。
    #     注: 对于 50 Hz 电力系统,时间窗Tw, 取 10 个额定基波周期,即为 200 ms。两条连续的频谱线之间的频率间隔是
    #     时间窗的倒数,因此两条连续的频谱线之间的频率间隔是 5 Hz。
    #     """
    # },
    # {
    #     "id": "Q8",
    #     "type": "技术参数",
    #     "question": "光伏发电系统接入配电网时无功功率调节的测试功率区间如何划分？",
    #     "answer": "按逆变器总额定功率的10%-40%、40%-60%、60%以上划分",
    #     "reference": """
    #     GB/T30152 6.1 无功容量
    #     检测应选择在晴天少云的气象条件下按照如下步骤进
    #     a)运行被测光伏发电系统，使其输出有功功率分别至光伏发电系统所配逆变器总额定功率的
    #     10% ~40%、40% ~60%、和60% 以上;
    #     b) 调节光伏发电系统功率因数从滞后0.95 开始,连续调节至超前 0.95;
    #     c) 测量并记录光伏发电系统实际输出的功率因数。
    #     """
    # }
]

# if __name__ == '__main__':
#     llm = load_llm()
#     JSONL_DIR = ''
#     OUT_DIR= ''
#
#     os.makedirs(OUT_DIR, exist_ok=True)
#     for JSONL_PATH, in os.listdir(JSONL_DIR):
#         file_name, file_ext = os.path.splitext(JSONL_PATH)
#         OUT_PATH = os.path.join(OUT_DIR, f"{file_name}_eval{file_ext}")
#         with open(JSONL_PATH, 'r', encoding='utf-8') as f, \
#              open(OUT_PATH, 'w', encoding='utf-8') as out_f:
#             for line in f:
#                 QA = json.loads(line)
#
#                 question = QA["问题"].strip().replace("\n", "")
#                 answer = QA["回答"].strip().replace("\n", "")
#                 standard = QA["原文"].strip().replace("\n", "")
#
#                 QA['评估'] = auto_evaluate(question, answer, standard, llm)
#                 out_f.write(json.dumps(QA,ensure_ascii=False) + '\n')
