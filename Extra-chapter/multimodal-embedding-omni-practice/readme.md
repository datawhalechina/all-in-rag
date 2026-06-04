# 多模态 Omni Embedding 实践（Jina v5-omni）

## 一、这个专题要解决什么问题

主线教程（`docs/chapter3/07_multimodal_embedding.md`）已经讲清楚了 CLIP 双编码器和 BGE-Visualized 图文检索的基础范式。本专题接着往前一步，验证下一代"统一多模态语义空间"的实际效果：

> **文本、图像、音频、视频、PDF 能不能被映射到同一个向量空间里，并按语义远近排序？**

本专题只做嵌入与相似度层面的验证，不做端到端 RAG pipeline。

**验收标准**：脚本跑通，输出一组相似度矩阵，并且你能说出"为什么海滩图像与海滩描述文字的相似度，高于与无关音频的相似度"。

**本专题不覆盖的内容**：跨模态检索的生产调优、大规模向量库接入、不同模型之间的性能对比。

---

## 二、技术演进：从 CLIP 到 v5-omni

理解本专题的实验前，有必要先看一眼这条技术路线是怎么走过来的：

| 阶段 | 模型代表 | 支持模态 | 核心思路 |
|------|----------|----------|----------|
| 第一代 | CLIP | 图 + 文 | 双编码器 + 对比学习，对齐图文向量空间 |
| 第二代 | BGE-Visualized-M3 | 图 + 文 | 在强文本底座上接入 patch token，联合建模 |
| 第三代 | Jina v5-omni | 图 + 文 + 音频 + 视频 + PDF | GELATO：冻结文本主干，仅训练轻量投影层 |

关键变化是：v5-omni 不需要重训整个模型就能扩展到新模态，而且已有文本向量的系统升级后**文本路径行为不变**，可以直接接入不需要重建全量索引。

---

## 三、GELATO 设计与向量空间

### 3.1 核心思路

GELATO（Geometry-preserving Embeddings via Locked Aligned TOwers）的核心是"冻结主干、只训练连接"：

- 文本主干（jina-embeddings-v5-text-nano）和非文本编码器均**冻结**
- 只训练投影层（`fc_vision_2`、`fc_audio`）和模态分隔符 token embedding
- 可训练参数约占总权重 **0.35%**
- 文本输入的 embedding 行为与原文本底座完全一致

这套设计让"支持新模态"和"保持已有文本几何稳定"可以在同一个模型里同时实现。

### 3.2 nano 版模型组成

| 组件 | 来源 | 参数量（加载视觉时） |
|------|------|------|
| 文本主干 | jina-embeddings-v5-text-nano | 0.24B |
| 视觉编码器 | SigLIP2 Base，来自 Qwen3.5-0.8B | 合计约 354M |
| 音频编码器 | Whisper-large-v3，来自 Qwen2.5-Omni-7B | 合计约 916M |
| 输出维度 | — | 768 维 |

加载时通过 `modality` 参数控制实例化哪些编码器塔，不需要的模态不会加载，节省显存。

### 3.3 任务适配器

模型内置 4 个任务适配器（LoRA + 任务专属投影权重），推理时通过 `default_task` 选择：

- `retrieval`：信息检索
- `text-matching`：语义相似度
- `clustering`：聚类
- `classification`：分类

本专题使用 `retrieval` 任务。

### 3.4 nano 与 small 的工程选型

两个版本均支持全部模态，差别主要体现在参数规模和 embedding 维度上。工程上通常先用 nano 跑通，再根据召回评测结果决定是否升级到 small。

---

## 四、实验场景

### 4.1 场景设定

假设你在做一个**海边内容检索**演示：语料库里有不同语言的文字描述、两张实拍图片、一段音频和一段视频，以及一份与海滩无关的学术论文摘录（作为"负样本"参照）。

目标是：用一个关于"海滩夕阳"的文本 query，验证不同模态的内容能否按语义远近正确排序。

### 4.2 data/ 文件说明

```
data/
├── beach1.jpg                        # 海滩夕阳实拍照片（约 50 KB）
├── beach2.jpg                        # 另一张海滩照片（约 78 KB）
├── example-audio-clip.wav            # 音频片段（约 4.6 MB）
├── example-video-clip.mp4            # 视频片段（约 968 KB）
└── paper_2506.18902_excerpt_2pages.pdf  # 论文摘录（与海滩无关，约 117 KB）
```

### 4.3 三轮 query 设计

脚本内置 3 轮语义相近但表述各异的 query，用于观察向量几何的稳定性：

| 轮次 | query 文本 |
|------|-----------|
| R1 | `sunset on the beach` |
| R2 | `waves and sunset on coast` |
| R3 | `beach scene with warm orange sky` |

---

## 五、目录结构

```
Extra-chapter/multimodal-embedding-omni-practice/
├── readme.md                         # 本文
├── images/
│   └── 3_2_3.png                     # v5-omni 架构图
├── code/
│   ├── 08_jina_embedding_omni.py     # 可运行脚本
│   └── pyproject.toml                # 依赖配置（uv）
└── data/
    ├── beach1.jpg
    ├── beach2.jpg
    ├── example-audio-clip.wav
    ├── example-video-clip.mp4
    └── paper_2506.18902_excerpt_2pages.pdf
```

---

## 六、最小实现骨架

下面是脚本主干的伪代码，对应 `code/08_jina_embedding_omni.py` 的执行顺序：

```python
# 1) 加载模型（本地优先，否则从 HF 下载）
raw_model = AutoModel.from_pretrained(model_path, default_task="retrieval", modality="vision")
processor = AutoProcessor.from_pretrained(model_path)
st_model  = SentenceTransformer(model_path, model_kwargs={"default_task": "retrieval"})
# raw_model + processor 用于文本和图像编码
# st_model 用于音频、视频、PDF 编码（自动按扩展名识别模态）

# 2) 构建语料库（corpus，9 个向量）
docs  = raw_model.embed(processor(text=[4 种语言描述], padding=True, ...))
img1  = raw_model.embed(processor(images=[beach1.jpg], text="<image>", ...))
img2  = raw_model.embed(processor(images=[beach2.jpg], text="<image>", ...))
audio = st_model.encode("example-audio-clip.wav")
video = st_model.encode("example-video-clip.mp4")
pdf   = st_model.encode("paper_2506.18902_excerpt_2pages.pdf")
corpus = stack([docs×4, img1, img2, audio, video, pdf])   # shape (9, 768)

# 3) 对每轮 query 编码并计算相似度矩阵
for name, q in [("R1", ...), ("R2", ...), ("R3", ...)]:
    qv     = raw_model.embed(processor(text=f"Query: {q}", ...))
    fusion = raw_model.embed(processor(images=[beach1.jpg], text=q, ...))
    # fusion = 同一张图 + 当前 query 文本，融合为单一向量
    vectors = stack([qv, corpus, fusion])   # shape (11, 768)
    sim = cosine_similarity(vectors)         # shape (11, 11)
    print(sim)

# 4) 对比三轮 query 之间的向量方向对齐度
aligned = [cosine(R1[i], R2[i]) for i in range(11)]
```

关键设计点：

- 文档编码使用 `"Document: "` 前缀，query 使用 `"Query: "` 前缀（retrieval 任务的非对称路由）
- `fusion` 向量把图像内容和 query 文字合并成一个向量，通常比纯文字 query 相似度更高
- `st_model.encode()` 传入文件路径即可，SentenceTransformers >= 5.4 会按扩展名自动选择音频/视频/PDF 编解码路径

---

## 七、环境准备与运行

### 7.1 依赖安装

> ⚠️ 处理音频、视频、PDF 需要的依赖比纯文本场景多。`pyproject.toml` 已配置好所有必要包，直接用 `uv sync` 安装即可。

```bash
cd Extra-chapter/multimodal-embedding-omni-practice/code
uv venv
source .venv/bin/activate
uv sync
```

主要依赖说明：

| 包 | 用途 |
|----|------|
| `sentence-transformers>=5.4.0` | 多模态 encode 支持（5.4 起才有） |
| `av>=17.0.0` | 视频帧提取 |
| `librosa>=0.11.0` + `soundfile` | 音频解码 |
| `pypdf` + `pypdfium2` | PDF 解析与渲染 |
| `pillow` + `torchvision` | 图像处理 |

**模型说明**：脚本优先使用本地路径 `models/jina-embeddings-v5-omni-nano`（相对仓库根目录），若不存在则自动从 HuggingFace 下载 `jinaai/jina-embeddings-v5-omni-nano`。国内网络建议提前手动下载。

### 7.2 执行脚本

```bash
# 从 code/ 目录运行（路径解析基于脚本位置，不依赖 shell 工作目录）
python 08_jina_embedding_omni.py
```

脚本启动时会打印 `model_source=...` 确认使用的是本地还是远程模型。加载过程会显示进度条（`Loading weights`），属正常现象。

> 💡 **transformers 警告**：启动时可能出现 `[transformers] torch_dtype is deprecated! Use dtype instead!` 这是 transformers 库内部版本迁移的提示，不影响结果，可忽略。

---

## 八、结果解读（真实输出）

### 8.1 向量布局

每一轮（R1/R2/R3）输出一个 shape=(11, 768) 的矩阵，11 个向量的排列固定如下：

| idx | 类型 | 内容 |
|-----|------|------|
| 0 | query | 当前轮次的文字 query（如 "Query: sunset on the beach"） |
| 1 | doc_en | "Document: A beautiful sunset over the beach" |
| 2 | doc_fr | "Document: Un beau coucher de soleil sur la plage" |
| 3 | doc_zh | "Document: 海滩上美丽的日落" |
| 4 | doc_ja | "Document: 浜辺に沈む美しい夕日" |
| 5 | img1 | beach1.jpg |
| 6 | img2 | beach2.jpg |
| 7 | audio | example-audio-clip.wav |
| 8 | video | example-video-clip.mp4 |
| 9 | pdf | paper_2506.18902_excerpt_2pages.pdf |
| 10 | fusion | img1 + 当前 query 文字（图文融合向量） |

### 8.2 R1 相似度矩阵（真实输出）

以下是 R1（query: `sunset on the beach`）的真实运行结果中 **query 行**（第 0 行）的相似度值：

```
R1 | shape=(11, 768)

query 行 sim[0][j]：

  idx  类型        相似度    说明
  ---  --------  -------  ------------------------------------------
   10  fusion     0.9159   ★★★ img1 + query 融合向量，远超纯文本 query
    3  doc_zh     0.7372   ★★★ 中文描述（海滩上美丽的日落）
    2  doc_fr     0.7200   ★★  法文描述（同义）
    4  doc_ja     0.5841   ★★  日文描述（同义）
    6  img2       0.5921   ★★  第二张海滩照片（跨模态）
    5  img1       0.5836   ★★  第一张海滩照片（跨模态）
    7  audio      0.1201   ★   音频，弱正相关
    8  video      0.0632       视频，接近零
    9  pdf       -0.0014       无关论文，接近零（符合预期）
    1  doc_en    -0.0153   ⚠   英文描述（见下方说明）
```

**观察一：图文融合 >> 纯文字 query**

`fusion`（0.9159）远高于纯文字 query 自身与语料的最高相似度（0.7372）。这印证了"图像 + 文字"联合输入可以更精准地表达检索意图。如果你的检索场景能提供示例图片，把它和 query 文字合并编码是一个值得尝试的方向。

**观察二：跨语言对齐工作良好**

法文（0.72）、中文（0.7372）、日文（0.5841）的描述都与英文 query 有正相关，体现了模型的多语言能力。这些文本在彼此之间的相似度更高（法文 vs 中文：0.84，中文 vs 日文：0.80），说明同语义内容在向量空间中形成了稳定的语义簇。

**观察三：图像跨模态相似度合理（约 0.58）**

两张海滩照片与文字 query 的相似度在 0.58 左右，明显低于同语义文本（0.72-0.74），但远高于音频（0.12）和视频（0.06）。图片之间的相似度很高（img1 vs img2：0.9167），说明模型能识别同一场景的不同拍摄。

**观察四：音视频相似度分层**

音频（0.12）、视频（0.06）、PDF（≈0）依次递减，与"内容相关性"基本一致：音频可能含有海浪或环境音，视频画面内容与海滩相关，而论文与海滩完全无关。

### 8.3 多轮对齐分析（真实输出）

```
R1 -> R2
aligned=[0.7623 1.     1.     1.     1.     1.     1.     1.     1.     1.     0.0309]
best_corpus_idx=3, score=0.7372

R1 -> R3
aligned=[0.6056 1.     1.     1.     1.     1.     1.     1.     1.     1.     0.2969]
best_corpus_idx=3, score=0.7372
```

**aligned** 数组中：

- 索引 1-9（语料库向量）的对齐值均为 **1.0**——语料库在每轮中重新编码，但输入不变，所以向量完全一致，属于预期行为。
- 索引 0（query 向量）：R1 vs R2 = **0.7623**，R1 vs R3 = **0.6056**。这说明语义相近但表述不同的 query，在向量空间中的方向不完全相同，R3 的措辞与 R1 差别更大。
- 索引 10（fusion 向量）：R1 vs R2 = **0.03**，R1 vs R3 = **0.30**。fusion 把图像与 query 文字合并，query 文字变化后 fusion 向量变化幅度更大——这说明 fusion 对文字语义的敏感度比纯文字 query 还要高。

**best_corpus_idx=3** 在三轮中保持一致，指向 `doc_zh`（中文海滩描述），score=0.7372。

### 8.4 自检指引

> 💡 **如何判断结果是否正常？**
>
> 正常结果应满足：
> - `fusion` 相似度 > 所有纯文本 query 的相似度
> - 同语义多语言文本的相似度 > 图像相似度 > 音视频相似度
> - 无关 PDF 的相似度接近零（-0.1 ~ 0.1 之间）
> - img1 vs img2 相似度 > 0.85（同场景两张图）
>
> 如果上述关系颠倒或数值全部接近零，先检查"常见失败点"第 2 条（依赖版本）。

---

## 九、常见失败点

### 9.1 `FileNotFoundError: Missing local asset`

脚本启动时找不到 `data/` 目录下的文件。确认你是从 `code/` 目录运行脚本，且 `data/` 文件夹已包含所有样本文件。

### 9.2 `ImportError` 或 `ModuleNotFoundError`（音频/视频相关）

通常是因为旧版依赖或某个包未安装。常见原因：

- `sentence-transformers < 5.4.0`：多模态 encode 支持从 5.4 起才有，旧版调用 `encode(audio_path)` 会报错。
- `av` 未安装：视频帧提取依赖 `PyAV`，需通过 `uv sync` 从更新后的 `pyproject.toml` 安装。
- `librosa` / `soundfile` 未安装：音频解码依赖。

解决方法：删除旧环境重建。

```bash
rm -rf .venv && uv venv && uv sync
```

### 9.3 HuggingFace 下载超时

国内网络访问 `huggingface.co` 可能超时。建议提前设置镜像或手动下载模型：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
python 08_jina_embedding_omni.py
```

或通过 `huggingface-cli download jinaai/jina-embeddings-v5-omni-nano --local-dir models/jina-embeddings-v5-omni-nano` 预先下载到本地。

### 9.4 路径解析错误（模型或数据）

脚本通过 `Path(__file__).resolve().parent` 计算路径，以脚本文件的位置为基准，不依赖 shell 的工作目录。但如果你移动了 `code/` 或 `data/` 的相对位置，需要同步修改脚本顶部的路径常量。

### 9.5 内存不足（OOM）

`v5-omni-nano` 在 MPS（Apple Silicon）上内存占用约 2-4 GB，正常可运行。如果遇到 OOM：

- 确认没有其他大模型占用内存
- 关闭浏览器等高内存占用程序
- 如果 MPS 不可用，脚本会自动退回 CPU（速度慢但可运行）

> ⚠️ **PDF 嵌入的额外内存开销**：对 PDF 做嵌入时，`pypdfium2` 会将每一页渲染成高分辨率图像再送入视觉编码器。页数多或分辨率高的 PDF 会在渲染阶段短暂占用大量内存（以及相应的 MPS 显存）。如果脚本在处理 PDF 时崩溃，优先排查是否内存不足，而不是代码问题。临时解决方案：在运行前关闭其他占内存的应用，或换用页数更少的 PDF 样本。

### 9.6 输出全部接近零

如果相似度矩阵里大多数值都在 ±0.01 之间，通常是模型加载时 `default_task` 或 `modality` 参数没有生效，导致向量没有走 retrieval 任务路径。确认 `transformers` 版本 >= 4.52 且 `sentence-transformers` >= 5.4.0。

---

## 十、局限与适用边界

1. **数据规模小**：本 demo 语料库只有 9 个向量，结论不代表生产环境的检索效果。
2. **视频和音频评测不充分**：各只有一个样本，时序理解能力无法验证。
3. **跨域迁移**：跨模态相似度受数据域影响明显，本 demo 的相似度数值（如音频 0.12、视频 0.06）不适用于其他类型的音视频内容。
4. **nano 模型局限**：观察到的英文文档异常（见 8.2）是 nano 模型的特定行为，small 模型或其他 API 接口可能有不同表现。
5. **模型和 API 持续演进**：文档中的参数上限与可用能力以官方最新页面为准。

---

## 十一、Gemini Embedding 2 参考

作为同类产品的参照，`gemini-embedding-2`（Vertex AI）也提供原生多模态嵌入能力：

- 支持文本、图像、文档、音频、视频的**交错输入**（interleaved）
- 默认输出维度 **3072**，可通过 `output_dimensionality` 下调
- 多模态共享上下文窗口（总 token 上限 8192）
- 各模态有独立的输入上限（图片数量、PDF 页数、音视频时长等）

与 v5-omni 的主要差异在于：Gemini Embedding 2 是托管 API，不支持本地部署；v5-omni 是开源模型，可本地运行，且能平滑兼容已有的 jina v5-text 文本索引。本专题引用此信息仅供横向理解，不做性能优劣判断。

---

## 十二、参考资料

- Jina v5-omni 官方说明：<https://jina.ai/news/jina-embeddings-v5-omni-multimodal-embeddings-for-text-image-audio-and-video/>
- GELATO 论文（arXiv 2605.08384）：<https://arxiv.org/abs/2605.08384>
- SentenceTransformers 5.4 多模态发布说明：<https://github.com/UKPLab/sentence-transformers/releases/tag/v5.4.0>
- Gemini Embedding 2（Vertex AI 文档）：<https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings>
- Gemini Embedding 2（DeepMind 模型页）：<https://deepmind.google/models/gemini/embedding/>
- 主线教程参考：`docs/chapter3/07_multimodal_embedding.md`
