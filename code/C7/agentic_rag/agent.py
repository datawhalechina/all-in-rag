"""
Agentic RAG 核心实现 —— 基于 LangGraph 的自主检索Agent

核心架构：
    analyze_query → retrieve → critique → refine_query → retrieve (循环)
                                            ↓
                                        generate → END

与传统RAG的关键区别：
1. Agent 自主分析问题、制定检索计划
2. Agent 对检索结果进行质量评估（而非盲目使用）
3. Agent 基于失败反馈有针对性地改写查询
4. 多轮检索结果被累积和整合
"""

from typing import TypedDict, List, Annotated, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import operator
import json
import re


# ============================================================
# Agent 状态定义
# ============================================================
class AgentState(TypedDict):
    """
    Agentic RAG Agent 的状态定义。

    与传统RAG的"一次性输入输出"不同，这里的 state 会在多轮检索中
    持续更新，记录每一步的决策、结果和评估。这是 Agentic RAG 能够
    进行"反思"和"迭代"的基础。
    """

    # 原始用户问题（保持不变）
    original_query: str

    # 当前检索查询（可能在 refine 阶段被改写）
    current_query: str

    # 对话/推理历史
    messages: Annotated[List, operator.add]

    # 检索历史：记录每一轮的查询、工具、结果、评估
    retrieval_history: List[dict]

    # 当前已收集的所有文档内容（去重合并）
    collected_docs: List[str]

    # 已见文档的hash，用于去重
    seen_doc_hashes: List[str]

    # 当前检索轮次
    iteration: int

    # 最大检索轮次（防止无限循环）
    max_iterations: int

    # 质量评估结果
    critique_result: dict

    # 推荐的检索工具列表
    recommended_tools: List[str]

    # 最终答案
    final_answer: str


# ============================================================
# AgenticRAG 类
# ============================================================
class AgenticRAG:
    """
    基于 LangGraph 的 Agentic RAG 实现。

    使用方式:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        agent = AgenticRAG(llm=llm, tools=[vector_search, web_search])
        result = agent.run("你的问题")
        print(result["answer"])
        print(result["trajectory"])  # 完整决策轨迹
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: List,
        max_iterations: int = 3,
        verbose: bool = True,
    ):
        """
        初始化 Agentic RAG。

        Args:
            llm: LLM 实例（用于推理、评估、生成）
            tools: Agent 可用的工具列表
            max_iterations: 最大检索轮次
            verbose: 是否打印详细日志
        """
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.graph = self._build_graph()

    # ----------------------------------------------------------
    # 图构建
    # ----------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态图"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("critique", self._critique)
        workflow.add_node("refine_query", self._refine_query)
        workflow.add_node("generate", self._generate)

        # 设置入口
        workflow.set_entry_point("analyze_query")

        # 定义边
        workflow.add_edge("analyze_query", "retrieve")
        workflow.add_edge("retrieve", "critique")

        # 条件边：根据评估结果决定下一步
        workflow.add_conditional_edges(
            "critique",
            self._should_continue,
            {
                "refine": "refine_query",
                "generate": "generate",
                "max_iter": END,
            },
        )

        workflow.add_edge("refine_query", "retrieve")
        workflow.add_edge("generate", END)

        return workflow.compile()

    # ----------------------------------------------------------
    # 节点1：查询分析
    # ----------------------------------------------------------
    def _analyze_query(self, state: AgentState) -> AgentState:
        """
        分析用户查询，制定检索策略。

        这是 Agentic RAG 区别于传统 RAG 的第一个关键步骤：
        - 传统 RAG 直接对原始查询进行检索
        - Agentic RAG 先分析问题类型、判断是否需要分解、选择初始检索策略
        """
        if self.verbose:
            print(f"\n{'─'*50}")
            print(f"🔍 [分析查询] 分析问题: {state['original_query'][:80]}...")

        analysis_prompt = f"""你是一个智能检索规划助手。分析以下用户问题并制定检索策略。

用户问题：{state['original_query']}

请判断：
1. 问题复杂度：simple（简单事实型）/ medium（需要一定推理）/ complex（多跳推理或需要多源信息）
2. 是否需要分解为子问题？如需，请列出子问题及其依赖关系
3. 推荐使用哪些检索工具？（可选: vector_search, web_search）
4. 初始检索查询应该是什么？（请优化措辞，使其更适合检索）

以 JSON 格式回复（只返回JSON，不要加markdown代码块标记）：
{{"complexity": "simple|medium|complex", "sub_queries": ["子问题1", "子问题2"], "dependencies": {{"子问题2": "子问题1"}}, "recommended_tools": ["vector_search"], "initial_query": "优化的初始检索查询"}}
"""
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])

        # 解析JSON（处理可能的markdown包裹）
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        try:
            plan = json.loads(content)
        except json.JSONDecodeError:
            if self.verbose:
                print(f"  ⚠️ JSON解析失败，使用默认策略。原始回复: {content[:200]}")
            plan = {
                "complexity": "medium",
                "sub_queries": [state["original_query"]],
                "dependencies": {},
                "recommended_tools": ["vector_search"],
                "initial_query": state["original_query"],
            }

        if self.verbose:
            print(f"  复杂度: {plan.get('complexity')}")
            print(f"  子问题: {plan.get('sub_queries')}")
            print(f"  推荐工具: {plan.get('recommended_tools')}")
            print(f"  初始查询: {plan.get('initial_query')}")

        state["current_query"] = plan.get("initial_query", state["original_query"])
        state["recommended_tools"] = plan.get("recommended_tools", ["vector_search"])
        state["retrieval_history"] = [
            {"step": "planning", "plan": plan, "original_query": state["original_query"]}
        ]

        return state

    # ----------------------------------------------------------
    # 节点2：执行检索
    # ----------------------------------------------------------
    def _retrieve(self, state: AgentState) -> AgentState:
        """
        执行检索操作。

        Agent 使用当前查询调用推荐的工具进行检索。
        支持多工具并行调用，结果合并去重。
        """
        if self.verbose:
            print(f"\n{'─'*50}")
            print(f"🔎 [检索 Round {state['iteration'] + 1}] 查询: {state['current_query'][:80]}...")

        current_query = state["current_query"]
        retrieval_results = []

        for tool_name in state.get("recommended_tools", ["vector_search"]):
            tool = self.tools.get(tool_name)
            if tool is None:
                if self.verbose:
                    print(f"  ⚠️ 工具 '{tool_name}' 不可用，跳过")
                continue

            try:
                if tool_name == "web_search":
                    result = tool.invoke({"query": current_query})
                else:
                    result = tool.invoke(current_query)

                retrieval_results.append(
                    {"tool": tool_name, "query": current_query, "result": result}
                )
                if self.verbose:
                    result_preview = str(result)[:150].replace("\n", " ")
                    print(f"  ✅ {tool_name}: {result_preview}...")
            except Exception as e:
                retrieval_results.append(
                    {"tool": tool_name, "query": current_query, "error": str(e)}
                )
                if self.verbose:
                    print(f"  ❌ {tool_name} 调用失败: {e}")

        # 记录检索历史
        state["retrieval_history"].append(
            {
                "step": f"retrieval_round_{state['iteration'] + 1}",
                "query": current_query,
                "results": retrieval_results,
            }
        )
        state["iteration"] += 1

        # 收集并去重文档
        for r in retrieval_results:
            if "result" in r and r["result"]:
                result_str = str(r["result"])
                doc_hash = str(hash(result_str))
                if doc_hash not in state["seen_doc_hashes"]:
                    state["seen_doc_hashes"].append(doc_hash)
                    state["collected_docs"].append(result_str)

        if self.verbose:
            print(f"  本轮收集: {len([r for r in retrieval_results if 'result' in r])} 条结果")
            print(f"  累计文档数: {len(state['collected_docs'])}")

        return state

    # ----------------------------------------------------------
    # 节点3：检索质量评估
    # ----------------------------------------------------------
    def _critique(self, state: AgentState) -> AgentState:
        """
        评估检索质量 —— Agentic RAG 的核心能力。

        Agent 对刚获取的检索结果进行语义层面的质量评估：
        - 相关性：文档是否与问题相关？
        - 充分性：信息是否足够回答原问题？
        - 可靠性：信息来源是否可信？
        """
        if self.verbose:
            print(f"\n{'─'*50}")
            print(f"📊 [质量评估] 第 {state['iteration']} 轮检索后评估")

        # 获取最近一轮的检索结果
        recent_results = state["collected_docs"][-5:]  # 只看最近的5条

        critique_prompt = f"""你是一个严格的检索质量评估助手。评估以下检索结果的质量。

━━━ 用户原始问题 ━━━
{state['original_query']}

━━━ 当前检索轮次 ━━━
第 {state['iteration']} / {state['max_iterations']} 轮

━━━ 最近检索到的文档 ━━━
{chr(10).join(f"[{i+1}] {doc[:300]}..." for i, doc in enumerate(recent_results))}

请逐项评估：
1. **相关性 (0-10)**：这些文档与原始问题的相关程度如何？
2. **充分性 (0-10)**：基于所有已收集的信息，能否完整回答原始问题？
3. **是否充分 (true/false)**：信息是否足够生成高质量答案？（严格判断）
4. **缺失维度**：如果不够，缺少哪些关键信息？（列出具体方面）
5. **失败原因**：如果不够，原因是什么？（词汇不匹配 / 查询太宽 / 查询太窄 / 角度偏差 / 信息源不对）
6. **建议改写**：如果需要重试，建议的改写查询是什么？

以 JSON 格式回复（只返回JSON，不要加markdown代码块标记）：
{{"relevance_score": 0-10, "sufficiency_score": 0-10, "is_sufficient": true/false, "missing_aspects": ["缺失1", "缺失2"], "failure_reason": "失败原因", "suggested_rewrite": "改写后的查询"}}
"""
        response = self.llm.invoke([HumanMessage(content=critique_prompt)])

        # 解析JSON
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        try:
            critique = json.loads(content)
        except json.JSONDecodeError:
            if self.verbose:
                print(f"  ⚠️ JSON解析失败，默认认为信息充分。原始回复: {content[:200]}")
            critique = {
                "is_sufficient": True,
                "relevance_score": 7,
                "sufficiency_score": 7,
                "missing_aspects": [],
                "failure_reason": "",
                "suggested_rewrite": "",
            }

        if self.verbose:
            print(f"  相关性: {critique.get('relevance_score')}/10")
            print(f"  充分性: {critique.get('sufficiency_score')}/10")
            print(f"  是否充分: {'✅ 是' if critique.get('is_sufficient') else '❌ 否'}")
            if not critique.get("is_sufficient"):
                print(f"  缺失维度: {critique.get('missing_aspects')}")
                print(f"  失败原因: {critique.get('failure_reason')}")

        state["critique_result"] = critique
        state["retrieval_history"].append({"step": "critique", "critique": critique})

        return state

    # ----------------------------------------------------------
    # 条件判断：决定下一步
    # ----------------------------------------------------------
    def _should_continue(
        self, state: AgentState
    ) -> Literal["refine", "generate", "max_iter"]:
        """
        根据评估结果决定 Agent 的下一步行动。

        决策逻辑:
        1. 达到最大轮次 → 强制生成（即使信息不够）
        2. 信息充分 → 生成答案
        3. 充足性分数 >= 7 → 生成答案（宽松阈值）
        4. 其他情况 → 改写查询重试
        """
        critique = state.get("critique_result", {})

        # 达到最大迭代次数
        if state["iteration"] >= state.get("max_iterations", 3):
            if self.verbose:
                print(f"\n  ⏰ 达到最大迭代次数 ({state['max_iterations']})，强制生成答案")
            return "max_iter"

        # 信息充分
        if critique.get("is_sufficient", False):
            if self.verbose:
                print(f"\n  ✅ 信息充分，进入生成阶段")
            return "generate"

        # 充分性分数高
        if critique.get("sufficiency_score", 0) >= 7:
            if self.verbose:
                print(f"\n  ✅ 充分性分数达标 ({critique.get('sufficiency_score')}/10)，进入生成阶段")
            return "generate"

        # 需要改写重试
        if self.verbose:
            print(f"\n  🔄 信息不足，进入查询改写阶段")
        return "refine"

    # ----------------------------------------------------------
    # 节点4：改写查询
    # ----------------------------------------------------------
    def _refine_query(self, state: AgentState) -> AgentState:
        """
        基于检索失败反馈，有针对性地改写查询。

        与第4章查询改写的区别：
        - 第4章的改写是"盲猜"式（固定的变体生成策略）
        - 这里的改写是"诊断"式（根据critique的失败原因针对性地改）
        """
        if self.verbose:
            print(f"\n{'─'*50}")
            print(f"✏️  [查询改写] 基于失败反馈优化查询")

        critique = state.get("critique_result", {})
        failure_reason = critique.get("failure_reason", "信息不足")
        suggested = critique.get("suggested_rewrite", "")

        # 如果LLM已给出改写建议，优先使用
        if suggested and suggested != state["current_query"]:
            new_query = suggested
            if self.verbose:
                print(f"  使用评估阶段建议的改写: {new_query[:80]}...")
        else:
            # 否则让LLM基于失败原因生成改写
            refine_prompt = f"""你需要改进一个检索查询。原查询检索效果不佳。

原始查询：{state['current_query']}
失败原因：{failure_reason}

改进策略参考：
- 词汇不匹配 → 使用同义词、相关术语、领域惯用表述
- 查询太宽 → 增加时间范围、具体实体名、限定条件
- 查询太窄 → 减少限定条件、使用更通用的表述
- 角度偏差 → 调整查询的关键词侧重（从"是什么"改为"为什么/怎么做"等）
- 信息源不对 → 明确需要的文档类型或来源

请生成一个新的查询。只返回查询文本，不要加引号或解释。"""
            response = self.llm.invoke([HumanMessage(content=refine_prompt)])
            new_query = response.content.strip().strip('"').strip("'")

        if self.verbose:
            print(f"  {state['current_query'][:60]}...")
            print(f"  → {new_query[:80]}...")

        state["retrieval_history"].append(
            {
                "step": "refine",
                "original_query": state["current_query"],
                "failure_reason": failure_reason,
                "refined_query": new_query,
            }
        )

        state["current_query"] = new_query
        return state

    # ----------------------------------------------------------
    # 节点5：生成答案
    # ----------------------------------------------------------
    def _generate(self, state: AgentState) -> AgentState:
        """
        基于所有收集的信息生成最终答案。

        Agent 整合多轮检索的全部结果，生成连贯、完整、有引用的答案。
        如果信息不足（达到最大轮次后仍不够），会诚实告知。
        """
        if self.verbose:
            print(f"\n{'─'*50}")
            print(f"📝 [生成答案] 基于 {len(state['collected_docs'])} 条文档生成答案")

        is_forced = state["iteration"] >= state.get("max_iterations", 3) and not state.get(
            "critique_result", {}
        ).get("is_sufficient", False)

        generate_prompt = f"""你是一个严谨的智能问答助手。请基于以下检索到的信息回答用户问题。

━━━ 用户问题 ━━━
{state['original_query']}

━━━ 检索到的信息 ━━━
{chr(10).join(f"--- 文档 {i+1} ---{chr(10)}{doc}" for i, doc in enumerate(state['collected_docs']))}

━━━ 检索过程 ━━━
- 检索轮次: {state['iteration']}
- 最后一轮评估充分性: {state.get('critique_result', {}).get('sufficiency_score', 'N/A')}/10
{"- ⚠️ 注意：已达到最大检索轮次，以下信息可能不完整" if is_forced else ""}

请回答：
1. 给出完整、准确的答案
2. 如果某些信息不确定或可能不完整，请明确指出
3. 如果适用，标注信息来源（引用文档编号）
4. 如果信息确实不足以完全回答问题，诚实说明，不要编造
"""
        response = self.llm.invoke([HumanMessage(content=generate_prompt)])

        state["final_answer"] = response.content
        state["retrieval_history"].append(
            {"step": "generate", "answer": response.content}
        )

        if self.verbose:
            print(f"  答案长度: {len(response.content)} 字符")

        return state

    # ----------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------
    def run(self, query: str) -> dict:
        """
        运行 Agentic RAG pipeline。

        Args:
            query: 用户问题

        Returns:
            {
                "answer": str,           # 最终答案
                "trajectory": List[dict], # 完整决策轨迹
                "iterations": int,       # 检索轮次
                "collected_docs_count": int, # 收集的文档数
            }
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🚀 Agentic RAG 启动")
            print(f"📝 用户问题: {query}")
            print(f"{'='*60}")

        initial_state = AgentState(
            original_query=query,
            current_query="",
            messages=[],
            retrieval_history=[],
            collected_docs=[],
            seen_doc_hashes=[],
            iteration=0,
            max_iterations=self.max_iterations,
            critique_result={},
            recommended_tools=[],
            final_answer="",
        )

        result = self.graph.invoke(initial_state)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✅ Agentic RAG 完成")
            print(f"  总轮次: {result['iteration']}")
            print(f"  文档数: {len(result['collected_docs'])}")
            print(f"{'='*60}\n")

        return {
            "answer": result["final_answer"],
            "trajectory": result["retrieval_history"],
            "iterations": result["iteration"],
            "collected_docs_count": len(result["collected_docs"]),
        }

    def run_stream(self, query: str):
        """
        流式运行，逐步yield每个步骤的状态（用于UI展示）。
        """
        initial_state = AgentState(
            original_query=query,
            current_query="",
            messages=[],
            retrieval_history=[],
            collected_docs=[],
            seen_doc_hashes=[],
            iteration=0,
            max_iterations=self.max_iterations,
            critique_result={},
            recommended_tools=[],
            final_answer="",
        )

        for event in self.graph.stream(initial_state):
            yield event
