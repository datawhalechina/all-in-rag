# Agentic RAG 端到端实践

> 配套教程见[第七章 Agentic RAG](../../../docs/chapter7/21_agentic_rag.md)。

## 快速开始

从仓库根目录安装本节依赖。

```bash
cd code/C7/agentic_rag
python -m pip install -r requirements.txt
```

`OPENAI_API_KEY` 用于模型问答和生成向量。只设置此密钥时，演示从本地知识库执行向量检索。需要联网检索时，再设置 `TAVILY_API_KEY`。

PowerShell

```powershell
$env:OPENAI_API_KEY = "你的 OpenAI API Key"
# $env:TAVILY_API_KEY = "你的 Tavily API Key"  # 可选
python demo.py
```

Bash

```bash
export OPENAI_API_KEY="你的 OpenAI API Key"
# export TAVILY_API_KEY="你的 Tavily API Key"  # 可选
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
