import re

from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from llm.llm import load_llm, MyLLM
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
            r"整定值", r"设定值", r"额定值", r"p\.u\.", r"百分比", r"区间",
            r"上限", r"下限", r"不超过", r"不低于", r"大于", r"小于", r"多少"
        ],
        "操作流程": [
            r"步骤", r"流程", r"操作方法", r"顺序", r"操作规程",
            r"如何", r"怎么操作", r"实施步骤", r"测试顺序",
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


def extract_parameters(question):
    """提取电力参数关键词"""
    params = {
        "电压": ["V", "kV", "伏特"],
        "电流": ["A", "kA", "安培"],
        "时间": ["s", "ms", "秒"]
    }
    return [p for p, units in params.items() if any(u in question for u in units)]


def find_similar_cases(text):
    """简单案例匹配（实际应接入案例库）"""
    known_cases = {
        "防孤岛": "Case2023-15：某光伏电站防孤岛测试报告",
        "低电压穿越": "Case2024-02：风电场LVRT测试记录"
    }
    return next((v for k, v in known_cases.items() if k in text), "")


def generate_type_specific_prompt( inputs):
    """根据问题类型生成特化Prompt"""
    context = inputs["context"]
    question = inputs["question"]
    # params = extract_parameters(question)
    # cases = find_similar_cases(question[:100])

    # 各类型特化指令
    type_instructions = {
        "标准条款": [
            "1. 回答中必须注明所依据的标准编号与对应条款号（如：GB/T 14285-2022 第5.2.3条）。",
            "2. 引用条款内容不得改写原文，尤其是涉及技术参数、限制条件和判断逻辑。",
            "3. 如涉及多个标准，请明确每个标准的适用场景、优先级，并注明各条款出处。",
            "4. 若标准存在更新或替代关系，请优先引用最新版本并说明变更点。"
        ],
        "计算验证": [
            "1. 明确引用的计算公式来源，逐步列出计算步骤，严禁直接给出最终答案。",
            "2. 每一步计算需展示中间变量、公式与单位（如：I = U / Z，单位kA），并解释变量含义。",
            "3. 计算结果需保留至小数点后两位，并注明精度控制或舍入处理方式。",
            "4. 若涉及极值判断、容差范围或边界条件，需说明判断标准与来源依据。"
        ],
        "操作流程": [
            "1. 按编号（1. 2. 3. ...）列出完整操作流程，保持逻辑顺序明确。",
            "2. 每一步骤应包含操作动作、所用设备/参数要求、执行条件和必要的安全警示语。",
            "3. 明确各步骤间的逻辑关系（串行/并行），如存在互锁、联动机制，必须说明。",
            "4. 若存在分支或判断流程，需明确条件触发逻辑并逐步展开说明。"
        ],
        "技术参数": [
            "1. 所有参数必须注明标准值、允许误差范围、单位（如±5%，220V±10V）。",
            "2. 如参数受环境因素影响（如温度、海拔等），需列出修正系数或修正公式。",
            "3. 若相同参数在不同标准/设备中存在差异，应注明来源并说明推荐选择理由。",
            "4. 建议提供测量方法、采样频率或检测依据，提升可追溯性。"
        ],
        "设备选型": [
            "1. 回答中应明确设备型号、选型依据、选用标准，并注明适用电压等级与容量范围。",
            "2. 如存在选型推荐公式或典型配置，应详细列出并注明来源。",
            "3. 对关键设备（如断路器、互感器等）应说明主要技术指标（开断容量、精度等级等）。",
            "4. 若有特殊安装要求（如户内/户外、接线方式），需明确指出。"
        ],
        "故障诊断": [
            "1. 回答应包含可能的故障原因、触发条件及影响范围（如跳闸、误动等）。",
            "2. 分析每个原因对应的检测方式或判断依据，避免主观猜测。",
            "3. 若有推荐处理措施或标准流程，请注明操作建议与参考依据。",
            "4. 如故障与配置/参数相关，建议指出其联动关系及可能改进方向。"
        ],
        "控制策略": [
            "1. 回答应覆盖控制目标、策略原理、实现手段（如自动投切、协调控制等）。",
            "2. 对主要控制逻辑流程图或执行判据进行简明描述，并说明适用场景。",
            "3. 如涉及设定值或动作门槛，应标明计算方法或推荐范围。",
            "4. 若控制策略存在优化建议或案例经验，可适当附带简要说明。"
        ],
        "安全规范": [
            "1. 指出相关安规条款编号与适用范围，确保依据充分。",
            "2. 回答必须涵盖人身安全、电气安全与操作隔离三类最基本要素。",
            "3. 若存在作业票、两票制度等约束条件，应予以列明。",
            "4. 如规范涉及动态行为（如带电作业、临时接地），请说明条件限制与风险控制。"
        ]
    }

    def get_instruction_by_type(q_type: str) -> str:
        return "\n".join(type_instructions.get(q_type, []))

    # return f"""
    # 请你扮演一位具有深厚电力系统背景的智能助手，针对电网相关的技术规程、检测标准、控制规范等文档内容，进行**严谨、分步骤**的推理和问答。请务必严格依赖提供的上下文，不得编造内容。
    # 要求如下：
    # 1. **题型判断**：
    #    - 含选项编号(A/B/C/D)→单选题（优先级最高）
    #    - 含下划线或疑问句→填空题
    #    - 陈述句→判断题
    #    - 未明确题型→填空题
    # 2. **输出规范**：
    #    - 单选/判断题→<answer>...</answer>包裹
    #    - 填空题→直接输出答案（无需标签）
    #    - 无法判断→输出"无法判断"
    # 3. **电力特化要求**：
    #    - 安全规程必引GB/T标准条款（如GB/T 7671-2007、DL/T 666-2012等）
    #    - 参数题保留单位（kV/MW/Hz）
    #    - 保护定值题标注误差范围（±5%）
    #    - 无上下文时→说明"未提供相关标准条款"
    # 4. **多任务识别**：
    #    - 优先识别问题类型（故障诊断/检测规程/控制策略/设备选型/技术参数）
    #    - 按类型选择输出模板
    # 5. 必须在回答结尾引用你在本次回答使用到的来源标准编号并注明本次回答参考的条款号
    # 6. 对于本文内交叉引用，必须在引用处条款号前加上标准编号
    #     例：回答：设置电网模拟装置输出电压，模拟表3中的一种不对称故障类型，电压跌落点选取应满足15.1的要求
    #     本条款引用了同一标准GB/T36548内的15.1条款，所以回答时应该是
    #     回答：模拟GB/T36548中的表3设置电网模拟装置输出电压模拟三相对称故障，电压跌落点满足GB/T36548 15.1要求。
    # 7.不允许按你的理解对回答进行省略或者缩减，严格按照依据回答
    #     例：原文：采集电压跌落前3秒到恢复正常后6秒之间的储能系统测试点电压和电流。
    #     不能回答为：采集电压跌落前3秒到恢复正常后6秒的数据。
    #     原文：设置电网模拟装置的输出电压模拟表3中的一种不对称故障类型。
    #     不能回答为：设置不对称故障类型
    # ---
    # 权威依据：
    # {context}
    # ---
    # 请按照以下步骤进行推理并回答问题：
    # 1. **题型判断**：明确问题类型（单选/判断/填空/无法判断）
    # 2. **上下文检索**：在权威依据中查找相关条款、参数范围或操作规则、若找不到相关依据请自行推理出答案
    # 3. **标准引用**：自动引用回答所依据的标准条款
    # 4. **分步验证**：
    #    - 故障诊断→现象分析→保护动作→根因定位
    #    - 检测规程→标准引用→测试步骤→合规判断
    #    - 控制策略→目标函数→约束条件→优化算法
    #    - 设备选型→参数计算→型号匹配→经济性分析
    #    - 技术参数→公式推导→标准校验→整定建议
    # 5. **输出格式**：
    #    - 推理过程：
    #    - 参考依据：
    #    - 注意事项（如有）：
    #    - <answer>最终答案</answer>
    # 6. **请用纯文本格式回答**：不要包含Markdown、代码块等特殊格式
    # 7. **可能存在OCR识别错误**：请按你的理解改正
    #
    # 问题如下：
    # {question}
    # 请开始逐步推理并给出答案：
    # """
    return f"""
    请你扮演一位具有深厚电力系统背景的智能助手，针对电网相关的技术规程、检测标准、控制规范等文档内容，进行**严谨、分步骤**的推理和问答。请务必严格依赖提供的上下文，不得编造内容。
    要求如下：
    1. **题型判断**：
       - 含选项编号(A/B/C/D)→单选题（优先级最高）注意：仅有一个正确选项
       - 含下划线或疑问句→填空题
       - 陈述句→判断题
       - 其他→正常回答即可
    2. **电力特化要求**：
       - 安全规程必引GB/T标准条款（如GB/T 7671-2007、DL/T 666-2012等）
       - 参数题保留单位（kV/MW/Hz）
       - 保护定值题标注误差范围（±5%）
       - 无上下文时→说明"未提供相关标准条款"
    3. **多任务识别**：
       - 优先识别问题类型（故障诊断/检测规程/控制策略/设备选型/技术参数）
       - 按类型选择输出模板
    4. 必须在回答结尾引用你在本次回答使用到的来源标准编号并注明本次回答参考的条款号
    5. 对于本文内交叉引用，必须在引用处条款号前加上标准编号
        例：回答：设置电网模拟装置输出电压，模拟表3中的一种不对称故障类型，电压跌落点选取应满足15.1的要求
        本条款引用了同一标准GB/T36548内的15.1条款，所以回答时应该是
        回答：模拟GB/T36548中的表3设置电网模拟装置输出电压模拟三相对称故障，电压跌落点满足GB/T36548 15.1要求。
    6.不允许按你的理解对回答进行省略或者缩减，严格按照依据回答
        例：原文：采集电压跌落前3秒到恢复正常后6秒之间的储能系统测试点电压和电流。
        不能回答为：采集电压跌落前3秒到恢复正常后6秒的数据。
        原文：设置电网模拟装置的输出电压模拟表3中的一种不对称故障类型。
        不能回答为：设置不对称故障类型
    ---
    权威依据：
    {context}
    ---
    请按照以下步骤进行推理并回答问题：
    1. **题型判断**：明确问题类型（单选/判断/填空/一般问题）
    2. **上下文检索**：在权威依据中查找相关条款、参数范围或操作规则、若找不到相关依据请自行推理出答案
    3. **标准引用**：自动引用回答所依据的标准条款
    4. **分步验证**：
       - 故障诊断→现象分析→保护动作→根因定位
       - 检测规程→标准引用→测试步骤→合规判断
       - 控制策略→目标函数→约束条件→优化算法
       - 设备选型→参数计算→型号匹配→经济性分析
       - 技术参数→公式推导→标准校验→整定建议
    5. **输出格式**：
       - 推理过程：
       - 参考依据：
       - 注意事项（如有）：
       - <answer>最终答案</answer>
    6. **请用纯文本格式回答**：不要包含Markdown、代码块等特殊格式
    7. **可能存在OCR识别错误**：请按你的理解改正
    8. **推理强化**：
       - 输入解析：提取关键词（电压等级/设备类型等）
       - 知识检索：学会引用通过rag检索到的DL/T/GB等标准条款，并给出依据
       - 多步验证：多个推理步骤
       - 输出格式化：按类型选择输出模板
    9. **最终答案输出规范**：
    - 判断题→仅输出"正确"或"错误"（无需解释）
    - 单选题→仅输出一个选项字母（如"A"）注意：仅有一个正确选项
    - 填空题→仅输出缺失内容（含下划线）或正常回答疑问句
    - 请将最终答案置于<answer>和</answer>之间。
    问题如下：
    {question}
    请开始逐步推理并给出答案：
    """


def build_rag_chain():
    vectorstore, client = load_qdrant_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 3})


    llm = MyLLM(enable_thinking=True,presence_penalty=1.2)

    # 文档格式化处理器
    def format_doc(doc):
        content = doc.page_content
        metadata = doc.metadata
        return (
            f"【文档：{metadata.get('source', '未命名文档')}】\n"
            f"文档id：{metadata.get('doc_id', '未知')} | "
            f"标题编号：{metadata.get('chunk_id', '无')}\n"
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
            "context": "\n\n".join(format_doc(doc) for doc in full_docs),
            # "source_docs": docs,  # 保留完整文档对象
            "question": inputs["question"]["query"],
            "docs": full_docs
        }

    def enhance_prompt(inputs):
        context = inputs["context"]
        question = inputs["question"]
        docs = inputs["docs"]
        print(question)
        for doc in docs:
            print(doc)

        # 动态生成元信息
        question_type = classify_question(question)

        enhanced_template = generate_type_specific_prompt(
            inputs=inputs
        )
        # return PromptTemplate.from_template(enhanced_template).format(
        #     context=context,
        #     question=question
        # )
        return {
            "prompt": enhanced_template,
            "question_type": question_type,
            "docs": docs
        }

    # 构建处理链
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
                        f"{doc.page_content}"
                        for doc in x["docs"]
                    ]
                }
            }
    )
