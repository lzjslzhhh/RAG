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

将来：构建电网领域的专业词典，引入能够理解文档（段落，图表，数学公式）的模型结合ocr来保证知识库的正确搭建

2. 存入向量数据库
    1. 选用gte-multilingual-base模型进行embedding
    2. 存入qdrant数据库
3. 知识增强（RAG）
    1. 使用langchain框架结合Qwen3-8B进行嵌入召回＋生成
    2. 借助大模型对向量数据库的检索结果进行rerank
    3. 多源知识融合：结构化（表格）、非结构化（文档）、图（电网拓扑）
4. 电网大模型关键增强技术 
    1. 训练微调模型（如LoRA，propmt调优）
    2. CoT思维链
        1. 自洽性（Self-Consistency）【1】：<font style="color:rgb(0, 0, 0);">通过抽样生成一组不同的推理路径，然后通过选择这些路径中最一致的路径来确定最终答案。多条推理路线可以得出正确答案，不同路径之间的一致性越高，表明对解决方案的置信度越高。通过避免贪婪解码的限制，自洽性利用模型的内部可变性来改善推理结果。</font>
        2. <font style="color:rgb(0, 0, 0);">思维树框架（ToT）【2】：通过将推理过程构建为可能的思维步骤的分支树，进一步推动了推理。ToT 使模型能够同时探索多个推理路径。每个分支代表不同的推理路线，模型在决定最佳路径之前评估各种中间步骤。这种树结构允许更广泛地探索潜在的解决方案，提高模型解决需要更复杂或创造性推理的任务的能力。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/43058383/1753155505884-8585402f-456e-4e00-9182-4c716f02d896.png)

5. 评估指标设计

## 实验设计
各实验在上一步基础上进行改进

| 实验编号 | 内容 |
| --- | --- |
| A1 | 原始大模型回答（零提示） |
| A2 | 大模型思考模式回答 |
| A3 | 加入 RAG 知识库知识 |
| A4 | 定制prompt模板增强CoT  |
| A5 | 结合大模型进行rerank |


## 待解决问题
#### 对于规范技术类文档的识别，包括条款编号，数学公式，图表等等
#### RAG输出的可解释性与可信度量化
#### 极端场景的鲁棒性
## 下一步计划
#### **<font style="color:#222a35;">选取更好的文档识别方法，增强知识库内容质量，</font>**
#### **<font style="color:#222a35;">建立测试问答集，使用RLHF或LoRA对大模型进行微调</font>**
#### **<font style="color:#222a35;">选取合适模型提取各个chunk关键词，实现向量+关键词的混合检索</font>**
## 参考文献
【1】X. Wang, J. Wei, D. Schuurmans, et al., “ Self-Consistency Improves Chain of Thought Reasoning in Language Models,” preprint, arXiv:2203.11171 (2022).

【2】S. Yao, D. Yu, J. Zhao, et al., “Tree of Thoughts: Deliberate Problem Solving With Large Language Models,” Advances in Neural Information Processing Systems 36 (2024): 11809–11822.





