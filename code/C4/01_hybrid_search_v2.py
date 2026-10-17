import json
import os

# 使用国内 HuggingFace 镜像（必须在导入模型相关库之前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, AnnSearchRequest, RRFRanker

# ============================================================
# 1. 初始化设置
# ============================================================
# Collection 名称（相当于数据库中的一张表）
COLLECTION_NAME = "dragon_siglip_demo"

# Milvus 服务地址（服务器模式）
MILVUS_URI = "http://localhost:19530"

# 数据文件路径（JSON 格式，包含若干条龙的元数据）
DATA_PATH = "../../data/C4/metadata/dragon.json"

# 批量插入的批大小（本示例未用到，预留变量）
BATCH_SIZE = 50

# ============================================================
# 2. 自定义 SigLIP 嵌入函数类
# ============================================================
class SigLIPEmbeddingFunction:
    """
    自定义的混合嵌入函数：
      - 稠密向量：使用 SigLIP 文本编码器（语义匹配）
      - 稀疏向量：使用 TF-IDF（关键词匹配）
    模拟 BGE-M3 的"稠密 + 稀疏"双路输出，方便替换到同一个混合检索框架中。
    """

    def __init__(self, model_name="google/siglip-base-patch16-256-multilingual", device="cpu"):
        """
        初始化 SigLIP 嵌入函数
        Args:
            model_name: SigLIP 模型名称（这里使用多语言版本）
            device: 运行设备 ("cpu" 或 "cuda")
        """
        self.model_name = model_name
        self.device = device

        print(f"--> 正在加载 SigLIP 模型: {model_name}")
        # 从 HuggingFace 加载 SigLIP 模型和对应的处理器（processor 负责 tokenize / 图像预处理）
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()  # 推理模式，关闭 dropout 等训练相关行为

        # -------- 初始化 TF-IDF 作为稀疏向量生成器 --------
        # max_features=10000：限制词汇表大小，控制稀疏向量维度
        # stop_words='english'：去掉英文停用词
        # ngram_range=(1, 2)：同时考虑单词和二元词组（如 "red dragon"）
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        # 标记 TF-IDF 是否已经拟合过（必须先 fit 才能 transform）
        self.tfidf_fitted = False

        # -------- 获取文本编码器的输出维度 --------
        # 用一条 dummy 文本走一遍前向，拿到 pooler_output 的维度作为稠密向量维度
        with torch.no_grad():
            dummy_text = ["test"]
            inputs = self.processor(text=dummy_text, padding="max_length", return_tensors="pt")
            # 只取文本相关输入，去掉图像字段（pixel_values）
            outputs = self.model.text_model(**{k: v.to(device) for k, v in inputs.items() if k != 'pixel_values'})
            self.dense_dim = outputs.pooler_output.shape[-1]

        print(f"--> SigLIP 模型加载完成。密集向量维度: {self.dense_dim}")

    @property
    def dim(self):
        """返回维度信息，兼容原 BGE-M3 接口（BGE-M3 的 ef.dim['dense'] 与 ef.dim['sparse']）"""
        return {
            "dense": self.dense_dim,
            "sparse": self.tfidf_vectorizer.max_features if self.tfidf_fitted else 10000
        }

    def fit_sparse(self, docs):
        """拟合稀疏向量模型（TF-IDF 需要先统计词频，构建词汇表）"""
        print("--> 正在拟合 TF-IDF 模型...")
        self.tfidf_vectorizer.fit(docs)
        self.tfidf_fitted = True
        print(f"--> TF-IDF 模型拟合完成。词汇表大小: {len(self.tfidf_vectorizer.vocabulary_)}")

    def encode_text_dense(self, texts):
        """使用 SigLIP 文本编码器把文本编码为稠密向量"""
        if isinstance(texts, str):
            texts = [texts]

        dense_vectors = []
        batch_size = 8  # 减小 batch size 以控制内存占用

        with torch.no_grad():
            # 分批处理，避免一次性输入过长导致 OOM
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                # 用 processor 做 tokenize / padding / 截断
                inputs = self.processor(text=batch_texts, padding="max_length", truncation=True, return_tensors="pt")
                # 只取文本输入（排除 pixel_values）
                inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'pixel_values'}

                # 只走 SigLIP 的文本塔
                outputs = self.model.text_model(**inputs)
                embeddings = outputs.pooler_output  # 取 pooler 输出作为句子表示

                # L2 归一化：让内积等价于余弦相似度
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                dense_vectors.extend(embeddings.cpu().numpy())

        return np.array(dense_vectors)

    def encode_text_sparse(self, texts):
        """使用 TF-IDF 把文本编码为稀疏向量"""
        if not self.tfidf_fitted:
            raise ValueError("请先调用 fit_sparse() 方法拟合TF-IDF模型")

        if isinstance(texts, str):
            texts = [texts]

        # 输出 scipy 稀疏矩阵（CSR 格式）
        sparse_matrix = self.tfidf_vectorizer.transform(texts)
        return sparse_matrix

    def __call__(self, texts):
        """
        主调用方法，返回 {'dense': ..., 'sparse': ...}
        兼容 BGE-M3 的调用方式 ef(docs)
        """
        if isinstance(texts, str):
            texts = [texts]

        # 如果还没拟合稀疏模型，先用当前文本集拟合（这里只是简单做法，实际应由训练语料拟合）
        if not self.tfidf_fitted:
            self.fit_sparse(texts)

        dense_vectors = self.encode_text_dense(texts)
        sparse_vectors = self.encode_text_sparse(texts)

        return {
            "dense": dense_vectors,
            "sparse": sparse_vectors
        }

# ============================================================
# 3. 连接 Milvus 并初始化嵌入模型
# ============================================================
print(f"--> 正在连接到 Milvus: {MILVUS_URI}")
connections.connect(uri=MILVUS_URI)

print("--> 正在初始化 SigLIP 嵌入模型...")
# 实例化自定义的 SigLIP 嵌入函数（有 GPU 时可改为 "cuda"）
ef = SigLIPEmbeddingFunction(device="cpu")

# ============================================================
# 4. 创建 Collection
# ============================================================
milvus_client = MilvusClient(uri=MILVUS_URI)

# 若集合已存在，先删除（演示环境，避免旧数据干扰）
if milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在删除已存在的 Collection '{COLLECTION_NAME}'...")
    milvus_client.drop_collection(COLLECTION_NAME)

# -------- 定义 Schema（字段结构）--------
fields = [
    # 主键（自动生成的 VARCHAR）
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),              # 图片 ID
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),                # 图片路径
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),               # 标题
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),        # 描述
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),             # 分类
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),            # 地点
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),          # 环境
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),           # 稀疏向量（TF-IDF）
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])  # 稠密向量（SigLIP）
]

# -------- 创建 Collection 及索引 --------
if not milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在创建 Collection '{COLLECTION_NAME}'...")
    schema = CollectionSchema(fields, description="使用SigLIP的龙混合检索示例")
    # 强一致性：写入后立刻可见
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
    print("--> Collection 创建成功。")

    # ---- 5. 创建索引 ----
    print("--> 正在为新集合创建索引...")

    # 稀疏向量索引：倒排索引 + 内积
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    collection.create_index("sparse_vector", sparse_index)
    print("稀疏向量索引创建成功。")

    # 稠密向量索引：AUTOINDEX 自动选择 + 内积
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    collection.create_index("dense_vector", dense_index)
    print("密集向量索引创建成功。")

# 重新获取 Collection 句柄
collection = Collection(COLLECTION_NAME)

# ============================================================
# 6. 加载数据并插入
# ============================================================
# 搜索前必须将集合加载到内存
collection.load()
print(f"--> Collection '{COLLECTION_NAME}' 已加载到内存。")

if collection.is_empty:
    print(f"--> Collection 为空，开始插入数据...")

    # 数据文件检查
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")

    # 读取 JSON
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # docs 用于嵌入；metadata 保存原始结构
    docs, metadata = [], []
    for item in dataset:
        # 拼接用于嵌入的字段（可按需扩展）
        parts = [
            item.get('title', ''),
            item.get('description', ''),
            item.get('location', ''),
            item.get('environment', ''),
            # *item.get('combat_details', {}).get('combat_style', []),
            # *item.get('combat_details', {}).get('abilities_used', []),
            # item.get('scene_info', {}).get('time_of_day', '')
        ]
        docs.append(' '.join(filter(None, parts)))
        metadata.append(item)

    print(f"--> 数据加载完成，共 {len(docs)} 条。")

    # 一次性生成稠密 + 稀疏向量
    print("--> 正在生成向量嵌入...")
    embeddings = ef(docs)
    print("--> 向量生成完成。")

    print("--> 正在分批插入数据...")

    # 按字段抽出各列数据
    img_ids = [doc["img_id"] for doc in metadata]
    paths = [doc["path"] for doc in metadata]
    titles = [doc["title"] for doc in metadata]
    descriptions = [doc["description"] for doc in metadata]
    categories = [doc["category"] for doc in metadata]
    locations = [doc["location"] for doc in metadata]
    environments = [doc["environment"] for doc in metadata]

    # ---- 处理向量格式 ----
    # 稠密向量：SigLIP 输出 numpy，需要转 list
    dense_vectors = embeddings["dense"].tolist()

    # 稀疏向量：TF-IDF 输出 scipy CSR，需要转成 Milvus 接受的 {token_id: weight} 字典
    sparse_vectors = []
    sparse_matrix = embeddings["sparse"]
    for i in range(sparse_matrix.shape[0]):
        row = sparse_matrix.getrow(i)
        sparse_dict = {}
        for j in range(row.nnz):
            sparse_dict[row.indices[j]] = float(row.data[j])
        sparse_vectors.append(sparse_dict)

    # 插入数据（字段顺序与 Schema 中非主键字段一致）
    collection.insert([
        img_ids,
        paths,
        titles,
        descriptions,
        categories,
        locations,
        environments,
        sparse_vectors,
        dense_vectors
    ])

    # 持久化
    collection.flush()
    print(f"--> 数据插入完成，总数: {collection.num_entities}")
else:
    print(f"--> Collection 中已有 {collection.num_entities} 条数据，跳过插入。")

# ============================================================
# 7. 执行搜索
# ============================================================
search_query = "悬崖上的巨龙"
search_filter = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
top_k = 5

print(f"\n{'='*20} 开始混合搜索 {'='*20}")
print(f"查询: '{search_query}'")
print(f"过滤器: '{search_filter}'")

# ---- 7.1 生成查询向量 ----
query_embeddings = ef([search_query])

# 稠密向量
dense_vec = query_embeddings["dense"][0].tolist()

# 稀疏向量：把 scipy CSR 的一行转成 {token_id: weight} 字典
sparse_matrix = query_embeddings["sparse"]
sparse_row = sparse_matrix.getrow(0)
sparse_dict = {}
for j in range(sparse_row.nnz):
    sparse_dict[sparse_row.indices[j]] = float(sparse_row.data[j])

# -------- 打印向量信息，便于理解 ----
print("\n=== 向量信息 ===")
print(f"密集向量维度: {len(dense_vec)}")
print(f"密集向量前5个元素: {dense_vec[:5]}")
print(f"密集向量范数: {np.linalg.norm(dense_vec):.4f}")

print(f"\n稀疏向量维度: {sparse_matrix.shape[1]}")
print(f"稀疏向量非零元素数量: {sparse_row.nnz}")
print("稀疏向量前5个非零元素:")
for i, (idx, val) in enumerate(list(sparse_dict.items())[:5]):
    print(f"  - 索引: {idx}, 值: {val:.4f}")

# 稀疏度：非零元素占比（TF-IDF 通常非常稀疏）
density = (sparse_row.nnz / sparse_matrix.shape[1] * 100)
print(f"\n稀疏向量密度: {density:.8f}%")

# 搜索参数（与索引 metric_type 一致）
search_params = {"metric_type": "IP", "params": {}}

# ---- 7.2 单独执行稠密向量搜索（作为对照）----
print("\n--- [单独] 密集向量搜索结果 ---")
dense_results = collection.search(
    [dense_vec],
    anns_field="dense_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,                        # 元数据过滤
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

for i, hit in enumerate(dense_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# ---- 7.3 单独执行稀疏向量搜索（作为对照）----
print("\n--- [单独] 稀疏向量搜索结果 ---")
sparse_results = collection.search(
    [sparse_dict],                             # 注意这里传的是字典格式的稀疏向量
    anns_field="sparse_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

for i, hit in enumerate(sparse_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# ---- 7.4 执行混合搜索（稀疏 + 稠密）----
print("\n--- [混合] 稀疏+密集向量搜索结果 ---")

# RRF 融合器：不关心两路分数的量纲，只用排名做融合
# k=60 是常用默认平滑常数
rerank = RRFRanker(k=60)

# 构造两路独立的搜索请求
dense_req = AnnSearchRequest([dense_vec], "dense_vector", search_params, limit=top_k)
sparse_req = AnnSearchRequest([sparse_dict], "sparse_vector", search_params, limit=top_k)

# 执行混合搜索：Milvus 并行检索两条路径，然后用 RRF 融合结果
results = collection.hybrid_search(
    [sparse_req, dense_req],
    rerank=rerank,
    limit=top_k,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

# 打印最终融合结果
for i, hit in enumerate(results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# ============================================================
# 8. 清理资源
# ============================================================
# 从内存中释放 Collection
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")

# 删除 Collection（演示用，避免下次运行重复数据）
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")