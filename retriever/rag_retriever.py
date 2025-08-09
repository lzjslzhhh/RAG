import re

from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from llm.llm import load_llm
from retriever.rag_embedding import COLLECTION_NAME
from vectorstore.qdrant_store import load_qdrant_vectorstore


def classify_question(question):
    """电力标准问答多维分类器（V3.1）"""
    question = question.lower()

    # --- 定义类别关键词 ---
    keyword_map = {
        "标准条款": [
            r"条款", r"章节", r"条目", r"标准", r"规范", r"规定", r"要求",
            r"\bgb/t\b", r"\bdl/t\b", r"ieee", r"iec", r"第[一二三四五六七八九十\d]+[章条]", r"q/gdw"
        ],
        "技术参数": [
            r"参数", r"数值", r"范围", r"允许值", r"±", r"误差",
            r"电压", r"电流", r"频率", r"温度", r"时间", r"阻抗", r"谐波",
            r"整定值", r"设定值", r"额定值", r"p\.u\.", r"百分比",
            r"上限", r"下限", r"不超过", r"不低于", r"大于", r"小于"
        ],
        "操作流程": [
            r"步骤", r"流程", r"操作方法", r"顺序", r"操作规程",
            r"如何操作", r"怎么操作", r"实施步骤", r"测试顺序",
            r"第一步", r"最后一步", r"准备阶段", r"测试阶段", r"验收阶段"
        ],
        "设备选型": [
            r"装置", r"设备", r"选型", r"型号", r"容量", r"配置",
            r"断路器", r"互感器", r"保护设备", r"继电器", r"测控装置", r"接线方式", r"接线图"
        ],
        "计算验证": [
            r"计算", r"怎么算", r"推导", r"计算过程", r"公式", r"求解",
            r"动作时间", r"电流整定", r"短路电流", r"灵敏度", r"验证", r"校核", r"极值点"
        ],
        "故障诊断": [
            r"故障", r"异常", r"跳闸", r"报警", r"影响", r"误动", r"失灵",
            r"可能导致", r"问题分析", r"风险", r"隐患", r"运行异常", r"误差过大"
        ],
        "控制策略": [
            r"调压", r"控制策略", r"自动投切", r"无功补偿", r"分接", r"优化控制",
            r"控制逻辑", r"控制原则", r"响应方式", r"自动化", r"协调控制"
        ],
        "安全规范": [
            r"安全距离", r"接地保护", r"操作票", r"安规", r"人身安全", r"安全规范", r"防误", r"检修安全"
        ]
    }

    # --- 识别优先级 ---
    priority_order = [
        "标准条款", "计算验证", "操作流程",
        "技术参数", "设备选型", "故障诊断", "控制策略", "安全规范"
    ]

    for category in priority_order:
        patterns = keyword_map[category]
        if any(re.search(p, question) for p in patterns):
            return category

    return "常规咨询"


def generate_prompt_a3(inputs):
    context = inputs["context"]
    question = inputs["question"]

    return f"""
    请用纯文本格式回答，不要包含Markdown、代码块等特殊格式,可能存在ocr识别错误（如："整定值"可能被误识别为"设定值"）,请按照你的理解纠正：
    1. **题型判断**：
    - 陈述句→判断题
    - 含选项编号(A/B/C)或"下列哪项"→单选题
    - 含下划线"___"或疑问句→填空题
     不用输出判断题型、仅当作划分输出规范的依据、按从上到下的优先级判断
     2. **输出规范**：
    - 判断题→仅输出"正确"或"错误"（无需解释）
    - 选择题→仅输出选项字母（如"A"）仅有一个正确选项
    - 填空题→仅输出缺失内容（含下划线）或正常回答疑问句
    - 对于单选和判断、禁止输出解析或选项内容
    【权威依据】
    {context}
    请根据信息回答以下问题，仅给出回答即可，不需要给出思考过程：
    {question}
    """


def generate_prompt_a4(inputs):
    context = inputs["context"]
    question = inputs["question"]

    return f"""
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
    
    【权威依据】
    {context}
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
    问题如下：
    {question}
    请开始逐步推理并给出答案，严格遵循给出的根据不同问题类型的输出规范，仅给出回答即可，不需要给出思考过程：
    """


def build_rag_chain(isA3=True):
    vectorstore, client = load_qdrant_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 3})


    llm = load_llm(enable_thinking=True)

    # 文档格式化处理器
    def format_doc(doc):
        content = doc.page_content
        metadata = doc.metadata
        return (
            f"【{metadata.get('source', '未命名文档')}】\n"
            f"文档id：{metadata.get('doc_id', '未知')} | "
            f"条款：{metadata.get('chunk_id', '无')}\n"
            f"各层级标题：{metadata.get('hierarchy', '无')}\n"
            f"内容：{content[:500]}{'...' if len(content) > 500 else ''}"
        )

    def retrieve_docs(inputs):
        docs = retriever.invoke(inputs["question"]["query"])
        # 提取关键信息
        # question_type = classify_question(inputs["question"]["query"])
        print(docs)
        full_docs = []
        for doc in docs:
            point_id = doc.metadata["_id"]
            result = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[point_id],
                with_payload=True
            )[0]
            doc.metadata.update(result.payload)
            full_docs.append(doc)
        return {
            "context": "\n\n".join(format_doc(doc) for doc in docs),
            # "source_docs": docs,  # 保留完整文档对象
            "question": inputs["question"]["query"],
            "docs": full_docs
        }

    def enhance_prompt(inputs):
        context = inputs["context"]
        question = inputs["question"]
        docs = inputs["docs"]

        question_type = classify_question(question)
        if isA3:
            prompt_template = generate_prompt_a3(
                inputs=inputs
            )
        else:
            prompt_template = generate_prompt_a4(
                inputs=inputs
            )

        return {
            "prompt": prompt_template,
            "question_type": question_type,
            "docs": docs
        }

    return (
            {"question": RunnablePassthrough()}  # 接收原始问题
            | RunnableLambda(retrieve_docs)  # 检索文档
            | RunnableLambda(enhance_prompt)  # 生成增强prompt
            | {
                "result": (lambda x: x["prompt"]) | llm,
                "metadata": lambda x: {
                    "type": x["question_type"],
                    "sources": [
                        f"{doc.metadata.get('doc_id', '未知')} {doc.metadata.get('source', '未知')} {doc.metadata.get('chunk_id', '无')}"
                        for doc in x["docs"]
                    ],
                    "contexts": [
                        doc.page_content for doc in x["docs"]
                    ]
                }
            }
    )
