import json
import os

# 使用国内 HuggingFace 镜像（必须在导入模型相关库之前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# ============================================================
# 1. 初始化设置
# ============================================================
# Collection 名称（相当于数据库中的一张表）
COLLECTION_NAME = "dragon_hybrid_demo"

# Milvus 服务的地址（这里使用服务器模式，需提前启动 Milvus）
MILVUS_URI = "http://localhost:19530"

# 数据文件路径（JSON 格式，包含若干条龙的元数据）
# 以脚本所在目录为基准，避免依赖运行时的当前工作目录
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../data/C4/metadata/dragon.json",
)

# 批量插入的批大小（本示例一次插入，预留变量方便扩展）
BATCH_SIZE = 50

# ============================================================
# 2. 连接 Milvus 并初始化嵌入模型
# ============================================================
print(f"--> 正在连接到 Milvus: {MILVUS_URI}")
# 建立与 Milvus 服务器的连接
connections.connect(uri=MILVUS_URI)

print("--> 正在初始化 BGE-M3 嵌入模型...")
# BGE-M3 是一个支持"稠密 + 稀疏"混合输出的多语言嵌入模型
# use_fp16=False：不使用半精度（CPU 上更稳定）
# device="cpu"：在 CPU 上推理
# 使用本地已下载的模型，避免每次联网下载
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../data/models/bge-m3",
)
ef = BGEM3EmbeddingFunction(model_name=MODEL_PATH, use_fp16=False, device="cpu")

# ef.dim["dense"] 表示稠密向量的维度（BGE-M3 通常为 1024）
print(f"--> 嵌入模型初始化完成。密集向量维度: {ef.dim['dense']}")

# ============================================================
# 3. 创建 Collection（若已存在则先删除，保证演示环境干净）
# ============================================================
milvus_client = MilvusClient(uri=MILVUS_URI)

# 如果集合已存在，先删除（演示用，避免旧数据干扰）
if milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在删除已存在的 Collection '{COLLECTION_NAME}'...")
    milvus_client.drop_collection(COLLECTION_NAME)

# -------- 定义 Schema（字段结构）--------
fields = [
    # 主键：自动生成的 VARCHAR（这里填 100 作为最大长度上限）
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    # 图片 ID
    FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
    # 图片路径
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),
    # 标题
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    # 描述（较长，4096 上限）
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
    # 分类（如 western_dragon / chinese_dragon / movie_character）
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    # 地点
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
    # 环境（如 mountain / cave 等）
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
    # 稀疏向量（用于 BM25 类的关键词匹配，由 BGE-M3 生成）
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    # 稠密向量（用于语义相似度检索，维度由 ef.dim["dense"] 决定）
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])
]

# -------- 创建 Collection 及索引 --------
if not milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在创建 Collection '{COLLECTION_NAME}'...")
    # 通过 Schema 定义表结构
    schema = CollectionSchema(fields, description="关于龙的混合检索示例")
    # consistency_level="Strong"：强一致性，写入后立即可查到（便于演示）
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
    print("--> Collection 创建成功。")

    # ---- 4. 创建索引 ----
    print("--> 正在为新集合创建索引...")

    # 稀疏向量索引：倒排索引，度量方式为内积（IP）
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    collection.create_index("sparse_vector", sparse_index)
    print("稀疏向量索引创建成功。")

    # 稠密向量索引：AUTOINDEX 让 Milvus 自动选择合适索引，度量方式为内积
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    collection.create_index("dense_vector", dense_index)
    print("密集向量索引创建成功。")

# 重新获取 Collection 句柄
collection = Collection(COLLECTION_NAME)

# ============================================================
# 5. 加载数据并插入
# ============================================================
# 将 Collection 加载到内存（搜索前必须 load）
collection.load()
print(f"--> Collection '{COLLECTION_NAME}' 已加载到内存。")

# 若集合为空则插入数据（避免重复插入）
if collection.is_empty:
    print(f"--> Collection 为空，开始插入数据...")

    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")

    # 读取 JSON 数据
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # docs 存放待嵌入的文本；metadata 保存原始数据
    docs, metadata = [], []
    for item in dataset:
        # 拼接用于嵌入的文本字段：标题 + 描述 + 地点 + 环境
        # （注释掉的部分是更多可用字段示例，可按需启用）
        parts = [
            item.get('title', ''),
            item.get('description', ''),
            item.get('location', ''),
            item.get('environment', ''),
            # *item.get('combat_details', {}).get('combat_style', []),
            # *item.get('combat_details', {}).get('abilities_used', []),
            # item.get('scene_info', {}).get('time_of_day', '')
        ]
        # 用空格拼接非空字段
        docs.append(' '.join(filter(None, parts)))
        metadata.append(item)

    print(f"--> 数据加载完成，共 {len(docs)} 条。")

    # 用 BGE-M3 一次生成所有文本的嵌入（返回 dict，含 sparse 与 dense）
    print("--> 正在生成向量嵌入...")
    embeddings = ef(docs)
    print("--> 向量生成完成。")

    print("--> 正在分批插入数据...")

    # 从 metadata 中按字段抽出各个列表（与 Schema 字段顺序一致）
    img_ids = [doc["img_id"] for doc in metadata]
    paths = [doc["path"] for doc in metadata]
    titles = [doc["title"] for doc in metadata]
    descriptions = [doc["description"] for doc in metadata]
    categories = [doc["category"] for doc in metadata]
    locations = [doc["location"] for doc in metadata]
    environments = [doc["environment"] for doc in metadata]

    # 获取稀疏向量和稠密向量
    sparse_vectors = embeddings["sparse"]
    dense_vectors = embeddings["dense"]

    # 批量插入数据（字段顺序需与 Schema 中非主键字段一致）
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

    # flush 确保数据持久化
    collection.flush()
    print(f"--> 数据插入完成，总数: {collection.num_entities}")
else:
    print(f"--> Collection 中已有 {collection.num_entities} 条数据，跳过插入。")

# ============================================================
# 6. 执行搜索
# ============================================================
# 查询语句
search_query = "悬崖上的巨龙"

# 过滤表达式：只在这些分类中检索
search_filter = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'

# 返回结果数量
top_k = 5

print(f"\n{'='*20} 开始混合搜索 {'='*20}")
print(f"查询: '{search_query}'")
print(f"过滤器: '{search_filter}'")

# ---- 6.1 生成查询向量 ----
# 注意：查询同样要用 BGE-M3 生成稀疏与稠密两种向量
query_embeddings = ef([search_query])
dense_vec = query_embeddings["dense"][0]                     # 稠密向量（numpy 数组）
sparse_vec = query_embeddings["sparse"]._getrow(0)           # 稀疏向量（scipy 稀疏矩阵的一行）

# -------- 打印向量信息，便于理解 ----
print("\n=== 向量信息 ===")
print(f"密集向量维度: {len(dense_vec)}")
print(f"密集向量前5个元素: {dense_vec[:5]}")
print(f"密集向量范数: {np.linalg.norm(dense_vec):.4f}")

print(f"\n稀疏向量维度: {sparse_vec.shape[1]}")
print(f"稀疏向量非零元素数量: {sparse_vec.nnz}")
print("稀疏向量前5个非零元素:")
for i in range(min(5, sparse_vec.nnz)):
    print(f"  - 索引: {sparse_vec.indices[i]}, 值: {sparse_vec.data[i]:.4f}")

# 稀疏度：非零元素占比（稀疏向量通常非常稀疏）
density = (sparse_vec.nnz / sparse_vec.shape[1] * 100)
print(f"\n稀疏向量密度: {density:.8f}%")

# 搜索参数：metric_type 必须与索引一致，使用内积 IP
search_params = {"metric_type": "IP", "params": {}}

# ---- 6.2 单独执行稠密向量搜索（作为对照）----
print("\n--- [单独] 密集向量搜索结果 ---")
dense_results = collection.search(
    [dense_vec],                              # 查询向量（此处为稠密）
    anns_field="dense_vector",                # 在稠密向量字段上搜索
    param=search_params,
    limit=top_k,
    expr=search_filter,                       # 元数据过滤
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]  # 只取第一个查询的结果

for i, hit in enumerate(dense_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# ---- 6.3 单独执行稀疏向量搜索（作为对照）----
print("\n--- [单独] 稀疏向量搜索结果 ---")
sparse_results = collection.search(
    [sparse_vec],                             # 查询向量（此处为稀疏）
    anns_field="sparse_vector",               # 在稀疏向量字段上搜索
    param=search_params,
    limit=top_k,
    expr=search_filter,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

for i, hit in enumerate(sparse_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# ---- 6.4 执行混合搜索（稀疏 + 稠密）----
print("\n--- [混合] 稀疏+密集向量搜索结果 ---")

# RRF（Reciprocal Rank Fusion）融合器：
# 不做分数加权，而是将两个检索器返回的排名做倒数求和，融合排名更稳健
# k=60 是 RRF 公式中的平滑常数，常用默认值
rerank = RRFRanker(k=60)

# 构造两个独立的搜索请求（稀疏一路、稠密一路）
# expr 过滤条件放在每个搜索请求上，保证与单路检索候选集一致
dense_req = AnnSearchRequest([dense_vec], "dense_vector", search_params, limit=top_k, expr=search_filter)
sparse_req = AnnSearchRequest([sparse_vec], "sparse_vector", search_params, limit=top_k, expr=search_filter)

# 执行混合搜索：Milvus 内部并行跑两条路径，再用 RRF 融合
results = collection.hybrid_search(
    [sparse_req, dense_req],                  # 多个搜索请求
    rerank=rerank,                            # 融合策略
    limit=top_k,                              # 最终返回条数
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

# 打印最终融合结果
for i, hit in enumerate(results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# ============================================================
# 7. 清理资源
# ============================================================
# 从内存中释放 Collection（释放内存占用）
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")

# 删除 Collection（演示用，避免下次运行数据重复）
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")