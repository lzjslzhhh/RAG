代码已开放在github[https://github.com/lzjslzhhh/RAG](https://github.com/lzjslzhhh/RAG)

## 问题分析
电网领域的知识具有以下特点：

+ 术语专业、概念多样（如“变电站”、“潮流计算”、“继电保护”）
+ 文献与标准结构复杂，长文本多
+ 实时性强（调度、故障诊断等场景）

目前存在的问题：

+ GPT-4 等大模型缺乏对结构化电网知识的深入理解
+ 回答缺乏推理过程或产生幻觉
+ 上下文不能长时间保留电网业务中的链式因果逻辑

## 解决思路
1. 技术文档识别
    1. 目前：简单使用ocr或pdfplumber结合正则表达式识别结构化文档

将来：构建电网领域的专业词典，引入能够理解文档（段落，图表，数学公式）的模型结合ocr或其他模型来保证知识库的正确搭建

2. 存入向量数据库
    1. 按照合适的策略对文本内容分块
    2. 选用gte-multilingual-base模型对文档内容按权重进行embedding
    3. 存入qdrant数据库
3. 知识增强（RAG）
    1. 使用langchain框架结合Qwen3-8B进行嵌入召回＋生成

将来：借助bge模型对向量数据库的检索结果进行rerank、多源知识融合：结构化（表格）、非结构化（文档）、图（电网拓扑）

4. 电网大模型关键增强技术 
    1. 训练微调模型（如LoRA等）
    2. prompt调优
        1. 高质量prompt核心要点：具体、丰富、少歧义
        2. 典型构成
            + 角色：给大模型一个最匹配任务的角色
            + 指示：对任务进行描述
            + 上下文：给出与任务相关的其他背景信息、尤其在多轮交互中（本任务中来自知识库）
            + 例子：必要时给出例子，学术中称为one-shot learning, few-shot learning或in-context learning
            + 输入：任务的输入信息；在prompt中明确的标识出输出
            + 输出：输出的格式描述，以便后续模块自动解析模型的输出结果（如JSON、XML）
        3. 调优思路
            1. 零样本提示
            2. 少样本提示
            3. 思维链（CoT）：通过中间推理步骤实现复杂的推理能力
            4. 少样本思维链（Few-shot CoT）
            5. 零样本CoT提示
    3. CoT思维链
        1. Multiverse让大模型自身并行进行推理，解决传统自回归大模型效率低的问题，同时能在许多任务上与传统大模型表现持平或更好
        2. 自洽性（Self-Consistency）【1】：通过抽样生成一组不同的推理路径，然后通过选择这些路径中最一致的路径来确定最终答案。多条推理路线可以得出正确答案，不同路径之间的一致性越高，表明对解决方案的置信度越高。通过避免贪婪解码的限制，自洽性利用模型的内部可变性来改善推理结果<font style="color:rgb(0, 0, 0);">。</font>
        3. 思维树框架（ToT）【2】：通过将推理过程构建为可能的思维步骤的分支树，进一步推动了推理。ToT 使模型能够同时探索多个推理路径。每个分支代表不同的推理路线，模型在决定最佳路径之前评估各种中间步骤。这种树结构允许更广泛地探索潜在的解决方案，提高模型解决需要更复杂或创造性推理的任务的能力。

![](https://cdn.nlark.com/yuque/0/2025/png/43058383/1753155505884-8585402f-456e-4e00-9182-4c716f02d896.png)

## 实验设计
各实验在上一步基础上进行改进

| 实验编号 | 内容 |
| --- | --- |
| A1 | 原始大模型回答（零提示） |
| A2 | 大模型思考模式回答 |
| A3 | 加入 RAG 知识库知识 |
| A4 | 定制prompt模板增强CoT  |
| A5 | 根据问题类型自动生成定制prompt |


A3引入RAG以在prompt里加入专业知识后、大模型对于Q1、Q2、Q3、Q4、Q5、Q8的回答效果有了明显提升，但对于Q6、Q7的提升效果有限，原因是按问题在向量数据库里检索到的Top-K个分块里没有包含答案、说明了引入rag存在提取到的上下文可能与答案无关

而A4在rag的基础上引入定制prompt模板后、大模型对于Q2、Q3、Q4的回答效果有了小幅提升、回答趋于专业化

A5中，再根据问题类型将对应回答要求和rag检索到上下文的标准号、所在章节号、标题动态加入prompt后、并加入few-shot learning、大模型针对各类型问题的回答更加细化，并能指出回答所具体使用的标准号，便于查证和进一步询问

## 待解决问题
1. RAG
    1. 构建索引时
        1. 文档中没有问题的答案
        2. 文档内容的准确性：比如如何从pdf中正确提取文字、表格等
        3. 文档分块的粒度：影响上下文完整性和输入给大模型的token数量
    2. 检索增强回答时
        1. 提取到的上下文与答案无关
        2. 只回答了一部分问题
        3. 上下文中有答案，但大模型没有提取出来
        4. 答案不够具体或过于具体
    3. 其他
        1. 知识库的健全与完整性要求

技术标准可能包括对其他技术标准的引用、例如“读取数字示波器数据进行分析,输出报表和测量曲线,并判别是否满足GB/T29319的要求,检测记录见附录A。”

        2. 知识碎片化问题

技术标准可能分为‘国标’、‘行标’、‘企标’，RAG返回多个检索结果时可能会出现冲突

        3. 知识库的实时性和有效性

对于规范技术类文档的正确识别，包括条款编号，数学公式，图表等等、对于废止的标准和新颁布的标准，要进行及时更新

## 下一步计划
1. 选取更好的文档识别方法，增强知识库内容质量，
2. 选取合适模型提取各个chunk关键词，实现向量+关键词的混合检索
3. 自动生成prompt

| 方向 | 文献 |
| --- | --- |
| 用蒙特卡洛树搜索（MCTS）策略，让模型自动生成接近专家级质量的 prompt，通过反复模拟和改进达成优化效果，适合在专业领域构建复杂 prompt 流程边迭代边优化  [4] | PromptAgent: Strategic Planning with Language Models Enables Expert‑level Prompt Optimization   |
| 研究发现 prompt 词汇具体性有最佳范围：既过通用也不宜过专，适合电网等 STEM 领域任务的 prompt 设计。提示词精炼度与专业用词精准度需要平衡  [5] | Prompt Engineering: How Prompt Vocabulary affects Domain Knowledge   |
| 用“受限 prompt”触发模型进行自训练，从而在无人工数据集情况下让模型逐渐定向专业领域，适合电网规范这种结构化高度要求的应用  [6] | From Manual Training to Domain‑Specific Adaptation through Constrained Prompt Engineering and Self‑Training   |


这些文献都强调：与其用固定模板，不如用自动/动态生成的 prompt，结合专业知识不断 refine，不仅更精准，还能自适应上下文变化，尤其适合电网这种高专业性、规范严格的领域。  

## 参考文献
【1】X. Wang, J. Wei, D. Schuurmans, et al., “ Self-Consistency Improves Chain of Thought Reasoning in Language Models,” preprint, arXiv:2203.11171 (2022).

【2】S. Yao, D. Yu, J. Zhao, et al., “Tree of Thoughts: Deliberate Problem Solving With Large Language Models,” Advances in Neural Information Processing Systems 36 (2024): 11809–11822.

【3】Z. Zhang, A. Zhang, M. Li, A. Smola“Automatic Chain of Thought Prompting in Large Language Models”preprint，arXiv:2210.03493（2022）

【4】 Wang, X., et al. (2023). _PromptAgent: Strategic planning with language models enables expert‑level prompt optimization_. arXiv. [https://arxiv.org/abs/2310.16427](https://arxiv.org/abs/2310.16427)

【5】 Schreiter, D. (2025). _Prompt Engineering: How prompt vocabulary affects domain knowledge_. arXiv. [https://arxiv.org/abs/2505.17037](https://arxiv.org/abs/2505.17037)

【6】 Chen, F. (2025). _From manual training to domain‑specific adaptation through constrained prompt engineering and self‑training_. Preprints. [https://www.preprints.org/manuscript/202504.1301/v1](https://www.preprints.org/manuscript/202504.1301/v1)







