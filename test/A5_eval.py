import asyncio
import json
import os
import re

from embedding.embedding import GTEEmbedding
from llm.llm import load_llm
from retriever.rag_evaluate import auto_evaluate_rag, evaluate_choice_metrics, evaluate_judge_metrics, \
    evaluate_fill_blanks
from eval_config import question_type
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 强制同步执行内核

# 启用设备端断言
os.environ['TORCH_USE_CUDA_DSA'] = '1'
if __name__ == '__main__':
    # queries = [
    #     '光伏发电系统接入配电网时如何进行防孤岛保护检测?',
    #     '电化学储能电站接入电网的额定能量如何进行测试?',
    #     '风力发电机在电网中的谐波电压适应性如何测试?'
    # ]
    # rag_chain = build_rag_chain(True)
    QAS = []
    y_true = []
    y_pred = []
    # embedding = GTEEmbedding()
    with open(f'./result/A5_{question_type}.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            question = json.loads(line)
            if question_type == "填空题":
                llm = load_llm(enable_thinking=False)
                evaluation = asyncio.run(evaluate_fill_blanks(question, llm))
                print(evaluation)
                result = question["result"]
                QAS.append({
                    '类型（单选、判断、判断）': question["type"],
                    '类型': result["metadata"]["type"],
                    '引用': result["metadata"]["sources"],
                    '上下文': result["metadata"]['contexts'],
                    '参考': question["reference"],
                    '问题': question["question"],
                    '标准答案': question["answer"],
                    '大模型回答': question["response"],
                    '相似度得分': evaluation["sim_score"],
                    '上下文召回率': evaluation["context_recall"],
                    '内容忠实度': evaluation["faithfulness"],
                    '事实正确度': evaluation["factual_correctness"],
                    '回答完整性': evaluation["answer_completeness"],
                    '表达清晰度': evaluation["clarity"]
                })
            else:
                y_true.append(question["answer"])
                pred = re.sub(r'<think>.*?</think>', '', question["response"], flags=re.DOTALL).split('.')[0]
                y_pred.append(pred)
        if question_type=="判断题":
            QAS.append({
                '题目数量':len(y_true),
                '评估':evaluate_judge_metrics(y_true, y_pred)
            })
        elif question_type=='单选题':
            QAS.append({
                '题目数量': len(y_true),
                '评估': evaluate_choice_metrics(y_true, y_pred)
            })
    with open(f'./result/A5_eval_{question_type}.json', 'w', encoding='utf-8') as f:
        for QA in QAS:
            f.write(json.dumps(QA, ensure_ascii=False) + '\n')
