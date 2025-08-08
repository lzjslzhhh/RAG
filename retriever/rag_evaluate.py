import asyncio
import re
from collections import Counter
from typing import Dict

import numpy as np
import torch.nn.functional as F
from ragas import EvaluationDataset, evaluate, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score
from transformers import AutoModel
from transformers import AutoTokenizer

from prompt_optimize.task import extract_answer, extract_judgment


def build_eval_prompt_answer_completeness(question, standard_answer, llm_answer, standard_text):
    return f"""
【任务说明】
你是电网行业的技术审查专家，具备国家标准和工程实践经验。请你根据权威依据和标准答案判断下列回答是否包含问题所需的全部关键信息点

问题：
{question}

权威依据：
{standard_text}

标准答案（关键词参考）：
{standard_answer}

模型生成回答（被评估对象）：
{llm_answer}

注意：
1.回答可能仅参考了部分检索到的依据，评估时注意这一点，若没有参考则说明没有检索到标准依据
2.请用纯文本格式回答: 不要包含Markdown、代码块等特殊格式。

请完成以下评估：
1. 回答是否完整覆盖上述所有关键点？
2. 漏掉了哪些要点？哪些信息是重复或冗余的？
3. 综合打分（0-100），其中100为完全覆盖，70-100表示覆盖多数，30-70表示覆盖一半左右，30以下为缺失严重。
"""


def build_eval_prompt_clarity(question, standard_answer, llm_answer, standard_text):
    return f"""
【任务说明】
你是电网行业的技术审查专家，具备国家标准和工程实践经验。请你根据权威依据和标准答案判断以下回答是否具有良好的逻辑结构和表述规范性

真实问题：
{question}

标准依据内容：
{standard_text}

标准答案（关键词参考）：
{standard_answer}

模型生成回答（被评估对象）：
{llm_answer}


注意：
1.回答可能仅参考了部分检索到的依据，评估时注意这一点，若没有参考则说明可能没有检索到标准依据（知识库不全或检索存在缺陷）
2.请用纯文本格式回答: 不要包含Markdown、代码块等特殊格式。


评估要点：
1. 特别对于故障诊断、操作流程、控制策略、安全规范类问题是否分步骤编号（如1. 2. 3.分步骤即可）？
2. 每步是否包含操作/参数/条件说明？
3. 是否使用标准术语（符合DL/T或GB/T）？
4. 单位是否标注完整（如kV、MW、Hz）？
5. 是否存在歧义、表达模糊、逻辑跳跃？

请逐项评估，并给出：
- 结构清晰度得分（0-100）
- 表达规范性得分（0-100）
- 优化建议（如有）
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


async def evaluate_fill_blanks(question,llm):
    answer_completeness = llm.invoke(
        build_eval_prompt_answer_completeness(question["question"], question["answer"], question["response"],
                                              question["contexts"]))
    clarity = llm.invoke(
        build_eval_prompt_clarity(question["question"], question["answer"], question["response"],
                                  question["contexts"]))
    print(answer_completeness, clarity)
    sim_score = sim_by_gte(question["answer"], question["response"])
    print(question["contexts"])
    evaluator_llm = LangchainLLMWrapper(llm)
    sample = SingleTurnSample(
        user_input=question["question"],
        reference=question["answer"],
        response=question["response"],
        reference_contexts=[question["reference"]],
        retrieved_contexts=question["contexts"],
    )
    context_recall = LLMContextRecall(llm=evaluator_llm)
    faithfulness = Faithfulness(llm=evaluator_llm)
    factual_correctness = FactualCorrectness(llm=evaluator_llm, atomicity='high', coverage='high')
    return {
        "sim_score": sim_score.item(),
        "context_recall": await context_recall.single_turn_ascore(sample),
        "faithfulness": await faithfulness.single_turn_ascore(sample),
        'factual_correctness': await factual_correctness.single_turn_ascore(sample),
        "answer_completeness": answer_completeness,
        "clarity": clarity,
    }


def evaluate_judge_metrics(y_true, y_pred, labels=None):
    if labels is None:
        labels = ["正确", "错误"]
    y_pred = [extract_judgment(y).strip('\n') for y in y_pred]
    print(y_pred[:5])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # # 危险操作专项统计（假设'错误'对应危险操作）
    # danger_mask = np.array(y_true) == '错误'
    # fn_danger = sum((np.array(y_pred) != np.array(y_true)) & danger_mask)
    # danger_recall = 1 - fn_danger / sum(danger_mask) if sum(danger_mask) > 0 else 1.0

    return {
        '准确率': np.mean(np.array(y_true) == np.array(y_pred)),
        # '召回率_危险操作': danger_recall,  # 电力特化指标
        '精确率_正确': precision[0],
        '召回率_正确': recall[0],
        'F1值': f1[0],
        # '误报率_安全操作': fn_danger / len(y_true)  # 误判安全操作比例
    }


def evaluate_choice_metrics(y_true, y_pred):

    y_pred = [extract_answer(y) for y in y_pred]
    print(y_pred[:5])
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # 干扰项分析
    wrong_choices: Dict[str, int] = dict(Counter([
        pred for true, pred in zip(y_true, y_pred) if true != pred
    ]))

    # 迷惑度指数计算
    entropy: float = 0.0
    if wrong_choices:
        total_wrong = sum(wrong_choices.values())
        probabilities = [count / total_wrong for count in wrong_choices.values()]
        entropy = -sum(p * np.log(p) for p in probabilities if p > 0)


    return {
            '准确率': accuracy,
            '精确率': precision,
            '召回率': recall,
            # '危险操作识别率': danger_recall,  # 电力特化指标
            '干扰项分布': dict(wrong_choices),
            '迷惑度指数': entropy
        }


def auto_evaluate_llm(question, llm, question_type):
    dataset = []
    dataset.append(
        {
            "user_input": question["question"],
            "retrieved_contexts": question["contexts"],
            "response": question["response"],
            "reference": question["answer"],
        }
    )
    print(dataset)
    if question_type == "填空题":
        return asyncio.run(evaluate_fill_blanks(question, llm))
    # elif question_type == '选择题':
    #     return evaluate_judge()


async def auto_evaluate_rag(question, llm, embedding):
    answer_completeness = llm.invoke(
        build_eval_prompt_answer_completeness(question["question"], question["answer"], question["response"],
                                              question["contexts"]))
    clarity = llm.invoke(
        build_eval_prompt_clarity(question["question"], question["answer"], question["response"],
                                  question["contexts"]))
    print(answer_completeness, clarity)
    dataset = []
    sim_score = sim_by_gte(question["answer"], question["response"])
    dataset.append(
        {
            "user_input": question["question"],
            "retrieved_contexts": question["contexts"],
            "response": question["response"],
            "reference": question["answer"],
        }
    )
    print(dataset)
    print(question["contexts"])
    evaluator_llm = LangchainLLMWrapper(llm)
    sample = SingleTurnSample(
        user_input=question["question"],
        reference=question["answer"],
        response=question["response"],
        reference_contexts=[question["reference"]],
        retrieved_contexts=question["contexts"],
    )
    context_recall = LLMContextRecall(llm=evaluator_llm)
    faithfulness = Faithfulness(llm=evaluator_llm)
    factual_correctness = FactualCorrectness(llm=evaluator_llm, atomicity='high', coverage='high')
    return {
        "sim_score": sim_score.item(),
        "context_recall": await context_recall.single_turn_ascore(sample),
        "faithfulness": await faithfulness.single_turn_ascore(sample),
        'factual_correctness': await factual_correctness.single_turn_ascore(sample),
        "answer_completeness": answer_completeness,
        "clarity": clarity,
    }


def auto_evaluate_rag_all(questions, llm, embedding):
    # llm_eval = llm.invoke(build_eval_prompt(question, standard_answer, llm_answer, standard_text), enable_thinking=True)
    # P,R,F1 = score([standard_answer],[llm_answer],lang='zh')
    dataset = []
    for question in questions:
        sim_score = sim_by_gte(question["answer"], question["response"])
        dataset.append(
            {
                "user_input": question["question"],
                "retrieved_contexts": question["contexts"],
                "response": question["response"],
                "reference": question["answer"],
            }
        )
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    evaluator_llm = LangchainLLMWrapper(llm)
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            # ContextRecall(),
            # Faithfulness(),
            FactualCorrectness()],
        llm=evaluator_llm,
        # raise_exceptions=False,  # 防止直接中断
        embeddings=embedding,
        # concurrency=1,
    )
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
    return result


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
    {
        "id": "Q2",
        "type": "计算验证",
        "question": "微电网接入配电网的低电压穿越测试中，三相电压跌落的极值点如何选取？",
        "answer": "应在UL1~UL2之间均匀选择7个点（含UL1和UL2），其中UL1为最低跌落点（如20%Un），UL2为正常工作电压下限（如85%Un）",
        "reference": """GB/T 34129-2017 7.5.3 d) 电压跌落点在 Ul1 一Ul2之间均匀选择 7 个跌落点且包括 2 个极值点 Ul1 和 Ul2 (Ul1 是低电穿越的电压最低跌落点,Up为正常工作电压的最低值,这两个值按照 GB/T 33589 的规定选取);"""
    },
    {
        "id": "Q3",
        "type": "操作流程",
        "question": "储能电站低电压故障穿越的空载测试步骤是什么？",
        "answer":
            """低电压故障穿越的空载测试按以下步骤进行，
                a) 断开被测储能系统与电网模拟装置之间的开关;
                b) 设置电网模拟装置的输出电压模拟线路三相对称故障，电压跌落点的选取应满足15.1的要求;
                c) 利用数据采集装置采集电压跌落前3 s到电压恢复正常后6Ss之间的储能系统测试点电压和电流 ，
                并记录;
                d) 重复b) ~c) ;
                e) 设置电网模拟装置的输出电压模拟表3中的一种不对称故障类型 ，电压跌落点的选取应满足15.1的要求
                f) 利用数据采集装置采集电压跌落前3 s到电压恢复正常后6s之间的储能系统测试点电压和电流 ，
                并记录;
                g) 重复e) ~f)
            """
        ,
        "reference": "GB/T 36548-2024 15.2.1 测试方法 a)~g)"
    },
    {
        "id": "Q4",
        "type": "技术参数",
        "question": "微电网接入配电网的并技术参数网测试若使用模拟电网需满足哪些参数要求？",
        "answer":
            """a)  可输出的短路电流应不小于被测微电网最大交换电流允许值的 20 倍;
            b) 谐波应小于 GB/T 14549 规定的谐波允许值的 50%;
            c) 稳态电压变化幅度不得超过额定电压的±1%;
            d) 电压偏差应在额定电压的±3%以内;
            e) 频率偏差应小于±0.01 Hz;
            f) 三相电压不平衡度应小于1% ,相位偏差应小于 1%;
            g) 中性点不接地的模拟电网,中性点位移电压应小于相电压额定值的1%;
            h) 具有在一个周波内进行±3%额定电压的调节能力;
            i) 具有在一个周波内进行±0.1%额定频率的调节能力;
            """
        ,
        "reference": """GB/T 34129-2017 5.2.3 并网测试用模拟电网要求
            a)  可输出的短路电流应不小于被测微电网最大交换电流允许值的 20 倍;
            b) 谐波应小于 GB/T 14549 规定的谐波允许值的 50%;
            c) 稳态电压变化幅度不得超过额定电压的±1%;
            d) 电压偏差应在额定电压的±3%以内;
            e) 频率偏差应小于±0.01 Hz;
            f) 三相电压不平衡度应小于1% ,相位偏差应小于 1%;
            g) 中性点不接地的模拟电网,中性点位移电压应小于相电压额定值的1%;
            h) 具有在一个周波内进行±3%额定电压的调节能力;
            i) 具有在一个周波内进行±0.1%额定频率的调节能力;"""
    },
    {
        "id": "Q5",
        "type": "标准条款",
        "question": "光伏发电系统电压适应性测试中，86%额定电压下的保持时间要求是多少？",
        "answer": "应保持1分钟",
        "reference": """
        GB/T30152 7.3 电压适应性
        检测应按照如下步骤进行:
        a) 在公共连接点标称频率条件下,调节电网模拟装置,使公共连接点电压至86%UN,并保持时间
        为1 min,记录光伏发电系统运行时间或脱网跳闸时间 ;
        b) 在公共连接点标称频率条件下 ,调节电网模拟装置,使公共连接点电压至 109%UN ,并保持时
        间为 1 min,记录光伏发电系统运行时间或脱网跳闸时间 ;
        c) 在公共连接点标称频率条件下,调节电网模拟装置,使公共连接点电压至 86 % Un ~ 109 %Un
        之间任意值 ,并保持时间为 1 min,记录光伏发电系统运行时间或脱网跳闸时间。
        注: UN 为公共连接点标称电压。
        """
    },
    {
        "id": "Q6",
        "type": "操作流程",
        "question": "光伏发电系统接入配电网时防孤岛保护检测时如何配置RLC负载？",
        "answer": "需满足：1)LC无功=系统无功 2)RLC有功=系统有功 3)品质因数1±0.2 4)流过K2的基波电流小于被测光伏发电系统输出电流的5%",
        "reference": """
        GB/T30152 10.3.2 检测步骤
        检测应按照如下步骤进行:
        a) 防孤岛能力检测点应选择在光伏发电系统并网点处。
        b) 通过功率检测装置测量被测光伏发电系统的有功功率和无功功率输出。
        c) 依次投入电感L、电容C 、电阻 R,使得:
            1) LC 消耗的无功功率等于被测光伏发电系统发出的无功功率;
            2) RLC 消耗的有功功率等于被测光伏发电系统发出的有功功率;
            3) RLC 谐振电路的品质因数为 1±0.2;
            4) 流过 K2 的基波电流小于被测光伏发电系统输出电流的5%。
        d) 断开K2, ,通过数字示波器记录被测光伏发电系统运行情况。
        e) 读取数字示波器和功率检测装置数据进行分析,若被测光伏发电系统在 2 s 内停止向交流负
        载供电,则不再继续检测 。否则应进行下步检测 。
        f) 调节电感L,电容C 使L、C 的无功功率按表 2 的规定每次变化±2%;表 2 中的参数表示的是
        图 2 中流经开关 K2 的无功功率流的方向,正号表示功率流从被测光伏发电系统到电网。
        g) 每次调节后 ,断开 K2, ,通过数字示波器记录被测光伏发电系统运行情况;若记录的时间呈持续
        上升趋势,则应继续以 2%的增量扩大调节范围,直至记录的时间呈下降趋势。
        h) 读取数字示波器数据进行分析,输出报表和测量曲线,并判别是否满足 GB/T 29319 的要求，
        检测记录见附录 A。
        """
    },
    {
        "id": "Q7",
        "type": "技术参数",
        "question": "光伏发电系统接入配电网时电压谐波测量时DFT频谱线间隔是多少？",
        "answer": "5Hz",
        "reference": """
        GB/T30152 3.4 时间窗 time window
        Tw
        测量电流谐波、间谐波所取的时间宽度。
        注: 对于 50 Hz 电力系统,时间窗Tw, 取 10 个额定基波周期,即为 200 ms。两条连续的频谱线之间的频率间隔是
        时间窗的倒数,因此两条连续的频谱线之间的频率间隔是 5 Hz。
        """
    },
    {
        "id": "Q8",
        "type": "技术参数",
        "question": "光伏发电系统接入配电网时无功功率调节的测试功率区间如何划分？",
        "answer": "按逆变器总额定功率的10%-40%、40%-60%、60%以上划分",
        "reference": """
        GB/T30152 6.1 无功容量
        检测应选择在晴天少云的气象条件下按照如下步骤进
        a)运行被测光伏发电系统，使其输出有功功率分别至光伏发电系统所配逆变器总额定功率的
        10% ~40%、40% ~60%、和60% 以上;
        b) 调节光伏发电系统功率因数从滞后0.95 开始,连续调节至超前 0.95;
        c) 测量并记录光伏发电系统实际输出的功率因数。
        """
    }
]
