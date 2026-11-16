# 层级式检索演示（RAPTOR 思路）

配套章节：`docs/chapter7/22_hierarchical_rag.md`

## 运行

```bash
python demo.py
```

需要 Python 3.8+，无第三方依赖、无模型下载、无 API Key。脚本用虚构笔记演示「全局叶子 top-k」与「先比较手工摘要，再在对应子树中检索」在同一查询预算下的差别。

## 适用范围

这是 RAPTOR 风格的结构演示，不是完整 RAPTOR：摘要由代码中的人工数据提供，分数使用 Python 标准库计算的中文字符二元组 Jaccard 重合度，不包含 SBERT、软聚类、LLM 摘要或生产级效果保证。更换语料或查询后，请重新运行并检查输出。

本机 Python 3.12.14 运行时，演示输出为：全局叶子 top-2 返回 `OPS-03`、`DEP-03`（Recall@2 = 0.0）；摘要引导检索返回 `OPS-03`、`OPS-02`（Recall@2 = 1.0）。这是单查询合成语料结果，不是通用性能结论。
