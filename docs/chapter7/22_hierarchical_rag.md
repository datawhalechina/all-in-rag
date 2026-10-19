# 第三节 层级式检索与长文档 RAG：从摘要树走到细节

> **前置知识**：建议先读第 3 章的向量检索与第 4 章的检索优化，再对照本章 [Graph RAG](./20_kg_rag.md) 和 [Agentic RAG](./21_agentic_rag.md)。本节以 RAPTOR 为代表，讲解把长文档组织成多层摘要节点的检索方法。

## 1. 什么时候需要层级式检索

传统 RAG 把文档切成段落或 token 窗口，为每个叶子 chunk 建索引。查询时直接在叶子里找 top-k。它擅长定位局部事实；遇到「这份手册的主要运维风险是什么」「几章内容共同说明了什么」这类主题或跨段落问题时，查询词未必与关键原文重合，top-k 可能只返回几个局部片段，缺少长文档结构。

RAPTOR（Recursive Abstractive Processing for Tree-Organized Retrieval）在叶子 chunk 之上递归构建摘要节点：把语义相关的节点聚类、让语言模型概括，再对摘要向量化并继续分层。查询时可从粗到细遍历树，或在折叠后的多层节点中统一选取上下文。因而检索器可以在「主题概览」和「原文细节」之间选择合适粒度，而不只返回同一种长度的片段。[RAPTOR 论文发表于 ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/8a2acd174940dbca361a6398a4f9df91-Paper-Conference.pdf)。

```text
原始文档
  └─ 叶子 chunk（保留原文与来源）
       └─ 语义聚类 + 摘要 → 中间层摘要
            └─ 再聚类 + 摘要 → 更高层主题

查询 → 选摘要层级 → 下钻或混合选择 → 返回摘要 / 原文证据
```

论文中的实现先用 SBERT 为约 100-token 的文本块生成向量，使用降维与软聚类把不同主题的文本组织起来，再让 GPT-3.5-turbo 生成父节点摘要，循环直到无法继续聚类。检索提供两种做法：**树遍历**逐层挑选相关节点，保留所走过的各层上下文；**折叠树**将不同层节点一起比较，在 token 预算内选节点。[论文方法与检索算法](https://proceedings.iclr.cc/paper_files/paper/2024/file/8a2acd174940dbca361a6398a4f9df91-Paper-Conference.pdf)。

## 2. 与 Graph RAG、Agentic RAG 怎么分工

| 方法 | 主要索引 / 决策结构 | 擅长解决的问题 | 主要代价与风险 |
| --- | --- | --- | --- |
| 叶子 chunk RAG（第 3–4 章） | 文本块向量或关键词索引 | 局部、明确的事实与段落定位 | 主题概括可能被切碎；结果粒度固定 |
| 层级式 RAG / RAPTOR | 原文叶子 + 多层摘要节点 | 长文档主题、跨段落关联、粗细粒度切换 | 建树要付出嵌入、聚类、摘要与刷新成本；摘要可能漏掉例外 |
| Graph RAG（§7.1） | 实体、关系、社区报告与原文 | 实体关系、多跳路径，以及基于社区报告的全局问题 | 抽取、实体对齐、图谱/社区维护和查询流程更复杂 |
| Agentic RAG（§7.2） | Agent 状态、检索工具与迭代轨迹 | 由查询决定是否检索、选择工具、反思结果并重试 | 多轮模型调用增加延迟、成本和调试复杂度 |

RAPTOR 的上层节点是文本摘要，不是实体关系边；它不能代替图上的关系遍历。Microsoft GraphRAG 会从文本抽取实体、关系与声明，构建社区并生成多层社区报告；其 local search 联合使用实体/图谱信息和原文 chunk，global search 则对社区报告做 map-reduce 汇总。[GraphRAG 索引概览](https://microsoft.github.io/graphrag/index/overview/) 与 [查询模式](https://microsoft.github.io/graphrag/query/overview/) 描述了这两类路径。

Agentic RAG 解决的是运行时的控制问题：Agent 根据问题动态决定使用什么工具、是否继续检索或改写查询。层级式索引解决的是资料本身的组织和检索粒度问题。一个系统可以让 Agent 先判断需要长文档主题检索，再调用层级检索器；但没有必要让每个简单问答都经过 Agent 循环。

**选型提示**：查一条产品参数，优先从叶子 chunk 起步；概括长手册的主题或关联分散章节，考虑加入摘要层级；追问实体间的明确关系，考虑 Graph RAG；需要在网页、知识库和数据库之间自主选择并多轮补证，才引入 Agentic RAG。

## 3. 从最小例子观察检索差异

示例代码位于 `code/C7/hierarchical_rag/`，使用完全虚构的三组运维笔记。一个宽泛问题询问「系统故障后怎么恢复数据服务并确认完整性？」。正确证据 `OPS-02` 描述暂停写入、加载快照、校验行数和抽样哈希、再切换备用实例；相邻的告警与备用入口段落虽然词面相似，但单独不足以支持完整答案。

在仓库根目录运行：

```bash
python code/C7/hierarchical_rag/demo.py
```

环境只需 Python 3.8 或更新版本；脚本只用标准库，不要安装向量库、模型或 API Key。示例为便于观察而**手工写好父层摘要**，用中文字符二元组 Jaccard 重合度代替 embedding；它演示「先选主题，再在主题内找原文」，不是 RAPTOR 的完整复现，也没有调用 LLM。

本次固定 `top_k=2`（两种方法都只给生成器两个叶子段落），脚本结果如下：

| 检索方式 | 返回叶子段落 | 命中人工标注的关键证据 OPS-02 | Recall@2 |
| --- | --- | --- | ---: |
| 全局叶子 top-k | OPS-03、DEP-03 | 否 | 0.0 |
| 摘要引导后在子树内检索 | OPS-03、OPS-02 | 是 | 1.0 |

结果解释：当前查询说的是整体故障恢复，但叶子检索先命中了「恢复演练」和「备用服务入口」；摘要节点把“数据完整性校验”和恢复主题聚到同一分支，使检索器再从这组叶子中找到真正执行步骤。此处只是一份手工构造的玩具语料和一个标注问题，分数是代码运行的检索结果，不代表生产数据上的准确率或 RAPTOR 的论文指标。更换查询、摘要或语料后，结果可能反转；请把它当作结构影响的可复现演示。

论文另在 QuALITY 测试集报告 RAPTOR + GPT-4 的准确率为 82.6%，表中 CoLISA 基线为 62.3%；这是论文整套系统与既有系统的结果对比，不应解释成只有检索器变化带来的纯净增益。论文也做了同一模型下的 controlled comparison：QuALITY 上统一使用 UnifiedQA-3B 时，SBERT + RAPTOR 为 56.6%，SBERT without RAPTOR 为 54.9%。这些数值来自论文数据集与模型设置，不能和上面这个 toy Recall@2 直接比较。[实验表格见论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/8a2acd174940dbca361a6398a4f9df91-Paper-Conference.pdf)。

## 4. 工程落地时要看住的边界

1. **父节点不能替代原文**：摘要压缩会丢失例外、版本号和条件。保留叶子及来源 ID；对金额、日期、权限、操作步骤等精确问题，应能回到原文核对。
2. **摘要是可追踪的数据产物**：存储输入节点 ID、摘要模型/提示版本、生成时间和文档版本。文档更新后，重新计算受影响的祖先节点；旧摘要不得悄悄继续代表新内容。
3. **摘要模型会引入错误**：RAPTOR 论文手工检查了 150 个摘要节点，其中 6 个（4%）含有某种幻觉，作者报告这些错误没有传播到更高层且对其 QA 任务没有可辨影响。这个小样本结果不能外推到所有模型、语料或业务；上线时仍需抽检摘要，并让最终回答引用叶子原文。[论文附录的幻觉分析](https://proceedings.iclr.cc/paper_files/paper/2024/file/8a2acd174940dbca361a6398a4f9df91-Paper-Conference.pdf)。
4. **建树不是免费的**：向量化、聚类、摘要生成会增加索引时成本；高频更新语料要比较增量更新与批量重建。摘要层级越多，检索路由的收益和维护面也越大。
5. **用评测集选择检索策略**：为本地业务准备主题型、关系型和精确事实型问题，标注证据 chunk，至少比较 Recall@k / Context Precision、上下文 token 数、延迟和建索引成本。若长文档主题问答占比很低，增加一棵摘要树可能不划算。

## 5. 与课程内容的连接

- 第 3 章提供 chunk、embedding 与向量索引基础，本节将索引单元从单层 chunk 扩展为不同抽象层级。
- 第 4 章介绍混合检索、查询构造与重写；层级检索可以和这些召回方式组合，最终仍要用证据 ID 评测。
- 本章 §7.1 的 Graph RAG 通过关系与社区结构组织证据；本节通过文本摘要树组织长文档结构。
- 本章 §7.2 的 Agentic RAG 能根据任务选择检索工具；层级检索可以作为其中一个工具，不承担 Agent 的规划与反思职责。

## 参考资料

1. Sarthi et al., [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/8a2acd174940dbca361a6398a4f9df91-Paper-Conference.pdf)，[arXiv 页面](https://arxiv.org/abs/2401.18059)，[作者实现](https://github.com/parthsarthi03/raptor)。
2. Microsoft, [GraphRAG Indexing Overview](https://microsoft.github.io/graphrag/index/overview/)，[Query Engine Overview](https://microsoft.github.io/graphrag/query/overview/)。
3. 本仓库：[基于知识图谱的 RAG](./20_kg_rag.md)、[Agentic RAG：让 RAG 具备自主决策能力](./21_agentic_rag.md)。
