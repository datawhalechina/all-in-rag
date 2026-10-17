# =============================================================================
# 多模态图文检索示例：Visualized-BGE + Milvus
# 功能：使用 Visualized-BGE 模型将图像和文本编码到同一向量空间，
#       存入 Milvus 向量数据库，并实现“以图搜图”或“图文混合检索”。
# =============================================================================

from dotenv import load_dotenv
load_dotenv()  # ⚠️ 必须放在最前面：加载 .env 文件中的环境变量（如 HF_ENDPOINT 镜像地址），确保后续模型下载/加载使用正确的源

import os
from tqdm import tqdm      # 进度条工具
from glob import glob      # 文件路径匹配
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE  # Visualized-BGE 多模态嵌入模型
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType  # Milvus 向量数据库客户端
import numpy as np
import cv2                 # OpenCV：用于图像处理与可视化拼接
from PIL import Image      # Pillow：用于图像读取与格式转换

# ========================= 1. 全局配置 =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录，保证相对路径在任何工作目录下都正确

MODEL_NAME = "BAAI/bge-base-en-v1.5"                  # BGE 文本基座模型名称（HuggingFace ID）
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "bge", "Visualized_base_en_v1.5.pth")  # 视觉对齐权重文件路径
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data", "C3")           # 数据集根目录
COLLECTION_NAME = "multimodal_demo"                     # Milvus Collection 名称
MILVUS_URI = "http://localhost:19530"                   # Milvus 服务地址


# ========================= 2. 核心工具类与函数 =========================

class Encoder:
    """
    多模态编码器封装类
    - 将纯图像、或"图像+文本"组合编码为统一维度的稠密向量
    - Visualized-BGE 通过训练将 CLIP 视觉特征对齐到 BGE 文本语义空间
    """
    def __init__(self, model_name: str, model_path: str):
        # 加载预训练的多模态 BGE 模型
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()  # 切换到推理模式（关闭 Dropout/BatchNorm 的训练行为）

    def encode_query(self, image_path: str, text: str) -> list[float]:
        """
        多模态查询编码：同时传入图像和文本
        适用场景：用户既提供了参考图片，又提供了文字描述来细化检索意图
        """
        with torch.no_grad():  # 推理时禁用梯度计算，节省显存并加速
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]  # 转为 Python list，取 batch 中第一条结果

    def encode_image(self, image_path: str) -> list[float]:
        """
        纯图像编码：仅传入图像路径
        适用场景：构建图像索引库时，对每张图库图像生成向量
        """
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]


def visualize_results(
    query_image_path: str,
    retrieved_images: list,
    img_height: int = 300,
    img_width: int = 300,
    row_count: int = 3
) -> np.ndarray:
    """
    将查询图和检索结果拼接为一张全景可视化图
    
    布局说明：
    ┌──────────┬─────────────────────────────┐
    │          │  Top0   │  Top1   │  Top2   │
    │  Query   ├─────────┼─────────┼─────────┤
    │ (红框)   │  Top3   │  Top4   │  ...    │
    │          ├─────────┼─────────┼─────────┤
    │          │  ...    │  ...    │  ...    │
    └──────────┴─────────────────────────────┘
    
    Args:
        query_image_path: 查询图像路径
        retrieved_images: 检索到的图像路径列表
        img_height/width: 每张缩略图的尺寸
        row_count: 结果区域的列数（行数自动根据结果数量计算）
    Returns:
        拼接后的 BGR numpy 数组，可直接用 cv2.imwrite 保存
    """
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    
    # 创建白色画布（BGR 格式，255=白色）
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    # 左侧查询图展示区域
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # --- 处理查询图像（左下角，红色边框标识）---
    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]  # RGB → BGR（OpenCV 格式）
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    # 添加红色边框，突出显示这是查询图
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    # 放置在左侧区域的最下方一行
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # --- 处理检索结果图像（右侧网格，黑色细边框+序号）---
    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count          # 计算在网格中的行列位置
        start_row, start_col = row * img_height, col * img_width  # 计算像素偏移
        
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        # 缩小 4px 留出边框空间
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved
        
        # 在左上角标注排名序号（蓝色）
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 水平拼接：左侧查询区 + 右侧结果区
    return np.hstack([query_display_area, panoramic_image])


# ========================= 3. 初始化编码器与 Milvus 客户端 =========================
print("--> 正在初始化编码器和Milvus客户端...")
encoder = Encoder(MODEL_NAME, MODEL_PATH)          # 加载多模态模型到 GPU/CPU
milvus_client = MilvusClient(uri=MILVUS_URI)       # 连接 Milvus 向量数据库


# ========================= 4. 创建 Milvus Collection =========================
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")

# 若已存在同名 Collection 则先删除（演示脚本保证幂等性）
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")

# 扫描数据目录下的所有 PNG 图像作为图库
image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))
if not image_list:
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")

# 动态获取向量维度（不同模型输出维度不同，避免硬编码）
dim = len(encoder.encode_image(image_list[0]))

# 定义 Collection 的字段结构
fields = [
    # 主键：自增 INT64，Milvus 自动生成唯一 ID
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # 向量字段：存储图像/文本的嵌入向量，维度与模型输出一致
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    # 标量字段：存储原始图像路径，检索后可回溯到原文件
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
]

# 构建 Schema 并创建 Collection
schema = CollectionSchema(fields, description="多模态图文检索")
print("Schema 结构:")
print(schema)

milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建 Collection: '{COLLECTION_NAME}'")
print("Collection 结构:")
print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))


# ========================= 5. 生成嵌入向量并插入数据 =========================
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")

data_to_insert = []
# 遍历所有图像，逐一编码为向量（生产环境建议批量编码以提升吞吐）
for image_path in tqdm(image_list, desc="生成图像嵌入"):
    vector = encoder.encode_image(image_path)
    data_to_insert.append({
        "vector": vector,         # 浮点向量
        "image_path": image_path  # 关联的原始文件路径
    })

if data_to_insert:
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']} 条数据。")


# ========================= 6. 创建向量索引并加载到内存 =========================
print(f"\n--> 正在为 '{COLLECTION_NAME}' 创建索引")

# 准备索引参数
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",           # HNSW：高性能近似最近邻索引，适合中小规模数据
    metric_type="COSINE",        # 余弦相似度：衡量方向相似性，对向量幅值不敏感
    params={
        "M": 16,                 # 每个节点的最大连接数，越大召回率越高但内存占用增加
        "efConstruction": 256    # 构建时的搜索宽度，越大索引质量越高但构建越慢
    }
)

milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
print("成功为向量字段创建 HNSW 索引。")
print("索引详情:")
print(milvus_client.describe_index(collection_name=COLLECTION_NAME, index_name="vector"))

# ⚠️ Milvus 要求显式加载 Collection 到内存后才能执行搜索
milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("已加载 Collection 到内存中。")


# ========================= 7. 执行多模态检索 =========================
print(f"\n--> 正在 '{COLLECTION_NAME}' 中执行检索")

query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
query_text = "一条龙"  # 文本补充描述，与图像共同构成多模态查询

# 使用"图像+文本"联合编码，融合两种模态的语义信息
query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)

# 执行向量近邻搜索
search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],              # 查询向量（支持批量，这里只查1条）
    output_fields=["image_path"],     # 返回结果中需要携带的标量字段
    limit=5,                          # 返回 Top-K 个最近邻
    search_params={
        "metric_type": "COSINE",      # 搜索时的距离度量（需与索引一致）
        "params": {"ef": 128}         # HNSW 搜索时的动态候选列表大小，越大越精确但越慢
    }
)[0]  # [0] 取第一个查询的结果列表

# 解析搜索结果
retrieved_images = []
print("检索结果:")
for i, hit in enumerate(search_results):
    print(f"  Top {i+1}: ID={hit['id']}, 距离={hit['distance']:.4f}, 路径='{hit['entity']['image_path']}'")
    retrieved_images.append(hit['entity']['image_path'])


# ========================= 8. 可视化结果 & 资源清理 =========================
print(f"\n--> 正在可视化结果并清理资源")

if not retrieved_images:
    print("没有检索到任何图像。")
else:
    # 拼接查询图与检索结果为全景图
    panoramic_image = visualize_results(query_image_path, retrieved_images)
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")
    cv2.imwrite(combined_image_path, panoramic_image)  # 保存到磁盘
    print(f"结果图像已保存到: {combined_image_path}")
    Image.open(combined_image_path).show()             # 弹窗预览

# 释放内存中的 Collection（不再占用 RAM，但数据仍在磁盘）
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")

# 删除 Collection（演示结束后清理，生产环境通常保留）
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")