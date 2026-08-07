"""
Agentic RAG 完整演示脚本

展示一条完整的 Agentic 检索-生成轨迹，包括：
1. 查询分析与规划
2. 多轮检索与工具选择
3. 检索质量自评
4. 基于反馈的查询改写
5. 答案生成

运行方式：
    python demo.py

前置条件：
    export OPENAI_API_KEY="your-key"
    export TAVILY_API_KEY="your-key"  # 可选，用于web搜索
"""

import sys
import os
import json

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from agent import AgenticRAG
from tools import VectorSearchTool, web_search_tool, build_knowledge_base


# ============================================================
# 演示知识库
# ============================================================
DEMO_DOCUMENTS = [
    # ── 2024年诺贝尔奖 ──
    (
        "2024年诺贝尔物理学奖授予John Hopfield和Geoffrey Hinton，"
        "以表彰他们在人工神经网络方面的基础性发现和发明。"
        "John Hopfield发明了一种能够存储和重建信息模式的网络结构（Hopfield网络）。"
        "Geoffrey Hinton发明了一种可以自主发现数据中属性的方法（玻尔兹曼机），"
        "对今天的大型人工神经网络至关重要。"
    ),
    (
        "2024年诺贝尔化学奖授予David Baker、Demis Hassabis和John Jumper。"
        "David Baker因计算蛋白质设计获奖，他领导的Rosetta项目成功设计出全新的蛋白质。"
        "Demis Hassabis和John Jumper因蛋白质结构预测获奖，"
        "他们开发的AlphaFold模型解决了困扰科学界50年的蛋白质折叠问题。"
    ),
    # ── AlphaFold与深度学习 ──
    (
        "AlphaFold是由DeepMind（Hassabis创立）开发的AI蛋白质结构预测系统。"
        "AlphaFold利用深度学习和神经网络技术，从氨基酸序列预测蛋白质的三维结构。"
        "AlphaFold2的核心架构包括：Evoformer模块（基于Transformer的进化信息处理）、"
        "结构模块（使用IPA——不变点注意力机制进行3D结构优化）。"
        "这些技术直接继承了深度学习领域的基础成果，包括注意力机制和反向传播算法。"
    ),
    (
        "AlphaFold的成功标志着AI for Science的重要里程碑。"
        "它将计算机科学的基础理论（神经网络、梯度下降、注意力机制）"
        "应用于生物学核心问题（蛋白质折叠），展示了跨学科研究的巨大潜力。"
        "2024年诺贝尔奖委员会明确指出，物理奖和化学奖共同揭示了一个科学范式转变："
        "AI不仅是一个工具，更成为了科学发现的驱动力。"
    ),
    # ── 深度学习基础 ──
    (
        "Geoffrey Hinton被称为'深度学习之父'。他的三大核心贡献："
        "1) 1986年发表反向传播算法论文（与Rumelhart、Williams合作），"
        "使多层神经网络的训练成为可能；"
        "2) 2006年提出深度信念网络（DBN）的逐层预训练方法，开启了深度学习时代；"
        "3) 2012年带领团队使用AlexNet在ImageNet竞赛中取得突破性成果，"
        "GPU算力+深度CNN+大规模数据的方法论成为行业标准。"
    ),
    (
        "Hopfield网络是一种内容可寻址的记忆系统，其核心思想是利用能量函数"
        "使网络能够收敛到存储的记忆状态。这一概念启发了后来的玻尔兹曼机、"
        "受限玻尔兹曼机（RBM）以及更广泛的能量基模型（Energy-Based Models）。"
        "Hopfield网络的能量最小化原理与蛋白质折叠的自由能最小化"
        "在数学上有深刻的类比关系。"
    ),
    (
        "反向传播算法（Backpropagation）是训练多层神经网络的核心算法。"
        "它通过链式法则计算损失函数对每个参数的梯度，实现高效的参数更新。"
        "这一算法是现代所有深度学习框架（PyTorch、TensorFlow、JAX）的基础，"
        "也是AlphaFold训练过程中优化数百万参数的核心数学工具。"
    ),
    # ── 补充背景 ──
    (
        "Demis Hassabis是DeepMind的联合创始人兼CEO。在创立DeepMind之前，"
        "他是一位国际象棋神童和游戏设计师。DeepMind的使命是'解决智能问题'"
        "（Solve Intelligence），然后用智能解决其他一切问题。"
        "AlphaFold是这一理念的典范：用通用AI技术解决特定科学问题。"
    ),
    (
        "David Baker是华盛顿大学的生物化学教授，也是Rosetta Commons的创始人。"
        "他的实验室开发了Rosetta软件套件，用于蛋白质结构预测和设计。"
        "Rosetta采用基于物理的能量函数和蒙特卡洛采样来探索蛋白质的构象空间。"
        "Baker团队还开发了Foldit游戏，让公众参与蛋白质折叠问题的解决。"
    ),
]


# ============================================================
# 测试问题集
# ============================================================
TEST_QUERIES = [
    # 核心测试问题：需要多跳推理的跨领域问题
    {
        "query": "2024年诺贝尔物理学奖得主的研究与2024年诺贝尔化学奖得主的研究之间有什么联系？",
        "description": "跨领域多跳推理",
        "expected_behavior": "需要至少2轮检索：先获取物理奖信息，再获取化学奖信息，然后分析技术联系",
    },
    # 简单问题：应该1轮就够
    {
        "query": "Geoffrey Hinton对深度学习有哪些主要贡献？",
        "description": "单跳事实性问题",
        "expected_behavior": "1轮检索即可获取足够信息",
    },
    # 复杂技术问题
    {
        "query": "Hopfield网络的能量函数概念与AlphaFold的蛋白质结构预测有什么技术上的关联？",
        "description": "深层技术关联分析",
        "expected_behavior": "需要多轮检索来获取Hopfield网络和AlphaFold的技术细节",
    },
]


# ============================================================
# 轨迹对比：传统RAG vs Agentic RAG
# ============================================================
def run_baseline_rag(query: str, vectorstore, llm) -> dict:
    """
    传统RAG的baseline实现（用于对比）。

    这是第1-5章教授的标准RAG pipeline：
    单次向量检索 → 拼接上下文 → LLM生成
    """
    docs = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""基于以下信息回答用户问题。

信息：
{context}

用户问题：{query}

请给出准确、完整的答案。如果信息不足，请指出。"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "retrieved_docs": len(docs),
        "context": context[:500],
    }


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("🧪 Agentic RAG vs 传统 RAG 对比演示")
    print("=" * 70)

    # ── 初始化 ──
    print("\n📦 初始化组件...")

    # 检查API密钥
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ 请设置 OPENAI_API_KEY 环境变量")
        print("   export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 构建知识库
    print("  构建向量知识库...")
    vectorstore = build_knowledge_base(
        documents=DEMO_DOCUMENTS,
        collection_name="agentic_rag_demo",
    )

    # 创建工具
    vs_tool_instance = VectorSearchTool(vectorstore)
    tools = [vs_tool_instance.search]

    # 如果有Tavily API密钥，添加web搜索
    has_web_search = bool(os.environ.get("TAVILY_API_KEY"))
    if has_web_search:
        tools.append(web_search_tool)
        print("  ✅ Web搜索工具已启用")
    else:
        print("  ℹ️  未设置 TAVILY_API_KEY，仅使用向量检索。设置后可启用web搜索。")

    # 创建Agent
    agent = AgenticRAG(llm=llm, tools=tools, max_iterations=3, verbose=True)

    # ── 运行测试 ──
    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]
        print(f"\n\n{'#'*70}")
        print(f"# 测试 {i+1}: {test['description']}")
        print(f"# 预期行为: {test['expected_behavior']}")
        print(f"{'#'*70}")

        # ── Agentic RAG ──
        print(f"\n{'▶'*20} Agentic RAG {'◀'*20}")
        agentic_result = agent.run(query)

        # ── 传统 RAG (Baseline) ──
        print(f"\n{'▶'*20} 传统 RAG (Baseline) {'◀'*20}")
        baseline_result = run_baseline_rag(query, vectorstore, llm)

        # ── 对比展示 ──
        print(f"\n{'─'*70}")
        print(f"📊 对比结果")
        print(f"{'─'*70}")

        print(f"\n{'┌' + '─'*30 + '┬' + '─'*20 + '┬' + '─'*20 + '┐'}")
        print(f"{'│ 指标':<30} {'│ Agentic RAG':<20} {'│ 传统 RAG':<20} │")
        print(f"{'├' + '─'*30 + '┼' + '─'*20 + '┼' + '─'*20 + '┤'}")
        print(f"{'│ 检索轮次':<30} {'│ ' + str(agentic_result['iterations']):<20} {'│ 1':<20} │")
        print(f"{'│ 收集文档数':<30} {'│ ' + str(agentic_result['collected_docs_count']):<20} {'│ ' + str(baseline_result['retrieved_docs']):<20} │")
        print(f"{'│ 答案长度(字符)':<30} {'│ ' + str(len(agentic_result['answer'])):<20} {'│ ' + str(len(baseline_result['answer'])):<20} │")
        print(f"{'└' + '─'*30 + '┴' + '─'*20 + '┴' + '─'*20 + '┘'}")

        # ── 轨迹摘要 ──
        print(f"\n📋 Agentic RAG 决策轨迹摘要:")
        for step in agentic_result["trajectory"]:
            step_name = step.get("step", "unknown")
            if step_name == "planning":
                plan = step.get("plan", {})
                print(f"  🔍 [规划] 复杂度={plan.get('complexity')}, "
                      f"子问题={plan.get('sub_queries')}")
            elif step_name.startswith("retrieval"):
                print(f"  🔎 [检索] query='{step.get('query', '')[:60]}...'")
            elif step_name == "critique":
                c = step.get("critique", {})
                print(f"  📊 [评估] 相关性={c.get('relevance_score')}/10, "
                      f"充分性={c.get('sufficiency_score')}/10, "
                      f"是否足够={'✅' if c.get('is_sufficient') else '❌'}")
                if not c.get("is_sufficient"):
                    print(f"         缺失: {c.get('missing_aspects')}")
                    print(f"         失败原因: {c.get('failure_reason')}")
            elif step_name == "refine":
                print(f"  ✏️  [改写] '{step.get('original_query', '')[:40]}...' → "
                      f"'{step.get('refined_query', '')[:40]}...'")
            elif step_name == "generate":
                print(f"  📝 [生成] 答案已生成 ({len(step.get('answer', ''))} 字符)")

        # ── 答案对比 ──
        print(f"\n{'─'*70}")
        print(f"✅ Agentic RAG 答案:")
        print(f"{'─'*70}")
        print(agentic_result["answer"][:800])

        print(f"\n{'─'*70}")
        print(f"✅ 传统 RAG 答案:")
        print(f"{'─'*70}")
        print(baseline_result["answer"][:800])

        if i < len(TEST_QUERIES) - 1:
            print(f"\n\n{'='*70}")
            input("按 Enter 继续下一个测试...")


if __name__ == "__main__":
    main()
