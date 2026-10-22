# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from unstructured.partition.auto import partition
# from unstructured.partition.pdf import partition_pdf
from dataclasses import field
import os
from glob import glob
from unittest import result

import cv2
import numpy as np
import torch
from PIL import Image
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from tqdm import tqdm
from visual_bge.visual_bge.modeling import Visualized_BGE

## 一、文本分割partition
# elements = partition(filename="/workspaces/all-in-rag/docs/chapter1/01_RAG_intro.md")

# print(elements)

# for i,elements in enumerate(elements):
# print(f"Element {i}: {elements}")

# elements_pdf = partition_pdf(filename="/workspaces/all-in-rag/docs/chapter1/test.pdf")

# print(elements_pdf)

# 二、向量数据库FAISS
# 1. 示例文本和嵌入模型
# text = [
#     "张三是法外狂徒",
#     "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
#     "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
# ]

# docs = [Document(page_content=t) for t in text]
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 2. 创建向量存储并保存到本地
# vectorstore = FAISS.from_documents(docs, embeddings)

# local_faiss_path = "./main/C1"
# vectorstore.save_local(local_faiss_path)

# print(f"FAISS index has been save to {local_faiss_path}")

# 3. 加载索引并执行查询
# 加载时需制定相同的嵌入模型，并允许序列化
# loaded_vectorstore = FAISS.load_local(
#     local_faiss_path, embeddings, allow_dangerous_deserialization=True
# )

# query = "FAISS是做什么的？"
# results = loaded_vectorstore.similarity_search(query, k=1)

# print(f"\n查询：'{query}'")
# print("相似度最高的文档：")
# for doc in results:
#     print(f"-{doc.page_content}")

# 三、向量数据库 Milvus
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "data/C3"
COLLECTION_NAME = "multimodal_demo"
MILVUS_URI = "./milvus_multimodal_demo.db"


class Encoder:
    """编码器类，用于将图像和文本编码为向量。"""

    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]


def visualize_results(
    query_image_path: str,
    retrieved_images: list,
    img_height: int = 300,
    img_width: int = 300,
    row_count: int = 3,
) -> np.ndarray:
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full(
        (panoramic_height, panoramic_width, 3), 255, dtype=np.uint8
    )
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    bordered_query = cv2.copyMakeBorder(
        resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0)
    )
    query_display_area[img_height * (row_count - 1) :, :] = cv2.resize(
        bordered_query, (img_width, img_height)
    )
    cv2.putText(
        query_display_area,
        "Query",
        (10, panoramic_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    # 处理检索到的文件
    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width
        
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved

        # 添加索引号
        cv2.putText(panoramic_image, str(i),(start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)

    return np.hstack([query_display_area, panoramic_image])

# 3. 初始化客户端
print("-->正在初始化编码器和milvus客户端...")
encoder = Encoder(MODEL_NAME, MODEL_PATH)
milvus_client = MilvusClient(uri=MILVUS_URI)

# 4. 创建 Milvus Collection
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的Collection：'{COLLECTION_NAME}'")

image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))
if not image_list:
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")
dim = len(encoder.encode_image(image_list[0]))

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length= 512),
]

# 创建集合 Schema
schema = CollectionSchema(fields, description="多模态图文检索")
print("Schema 结构：")
print(schema)

# 创建集合
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建Collection：'{COLLECTION_NAME}'")
print("Collection 结构")
print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))

#5. 准备数据
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")
data_to_insert = []
for image_path in tqdm(image_list, desc="生成图像嵌入"):
    vector = encoder.encode_image(image_path)
    data_to_insert.append({"vector": vector, "image_path": image_path})

if data_to_insert:
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']}条数据。")

# 6. 创建索引
print(f"\n--> 正在为'{COLLECTION_NAME}' 创建索引")
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16,"efConstruction": 256}
)
milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
print("成功为向量字段创建 HNSW 索引。")
print("索引详情:")
print(milvus_client.describe_index(collection_name=COLLECTION_NAME, index_name="vector"))
milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("已加载 Collection 到内存中。")

# 7. 执行多模态检索
print(f"\n--> 正在'{COLLECTION_NAME}' 中执行检索")
query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
query_text = "一条龙"
query_vector = encoder.encode_query(image_path=query_image_path,text=query_text)

search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    output_fields=["image_path"],
    limit=5,
    search_params={"metric_type":"COSINE", "params":{"ef": 128}}
)[0]

retrieved_images = []
print("检索结果：")
for i, hit in enumerate(search_results):
    print(f" Top {i+1}: ID={hit['id']}, 距离={hit['distance']:.4f},路径='{hit['entity']['image_path']}'")
    retrieved_images.append(hit['entity']['image_path'])

# 8.可视化与清理
if not retrieved_images:
    print("没有检索到任何图像。")
else:
    panoramic_image = visualize_results(query_image_path, retrieved_images)
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")
    cv2.imwrite(combined_image_path, panoramic_image)
    print(f"结果图像已保存到：{combined_image_path}")
    Image.open(combined_image_path).show()

milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放Collection：'{COLLECTION_NAME}'")
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除Collection：'{COLLECTION_NAME}'")