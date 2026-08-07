# Agentic RAG 端到端实践

> 配套教程章节：`docs/chapter7/21_agentic_rag.md`

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置API密钥
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="..."  # 可选，用于web搜索

# 3. 运行演示
python demo.py
```

## 文件结构

```
agentic_rag/
├── agent.py          # AgenticRAG 核心实现（LangGraph状态图）
├── tools.py          # 检索工具定义（向量检索 + Web搜索）
├── demo.py           # 完整演示：Agentic RAG vs 传统RAG对比
└── requirements.txt  # 依赖列表
```

## 核心特性

- **自主查询分析**：Agent分析问题类型后制定检索计划
- **多工具选择**：向量检索 + Web搜索，Agent自主选择
- **检索质量自评**：每次检索后评估充分性（0-10分）
- **诊断式改写**：基于失败原因针对性改写查询（非盲猜）
- **完整轨迹记录**：每一步的决策、结果、评估均可追溯

## Agent状态图

```
analyze_query → retrieve → critique → refine_query → retrieve
                          ↓
                      generate → END
```
