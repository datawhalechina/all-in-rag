from dotenv import load_dotenv
load_dotenv()  # 加载 .env（HF 镜像），必须在模型加载前

# 导入 FAISS 向量存储类，用于构建和操作 FAISS 索引
from langchain_community.vectorstores import FAISS
# 导入 HuggingFaceEmbeddings 类，用于加载 Hugging Face 的嵌入模型
from langchain_community.embeddings import HuggingFaceEmbeddings
# 导入 Document 类，用于将文本封装为 LangChain 文档对象
from langchain_core.documents import Document

# 1. 准备示例文本和嵌入模型
# 定义一组示例文本，用于构建向量索引
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]
# 将每个文本字符串转换为 Document 对象，便于 LangChain 处理
docs = [Document(page_content=t) for t in texts]
# 初始化 Hugging Face 嵌入模型，这里使用中文小模型 BAAI/bge-small-zh-v1.5
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 2. 创建向量存储并保存到本地
# 使用文档和嵌入模型构建 FAISS 向量存储（内部会计算嵌入并建立索引）
vectorstore = FAISS.from_documents(docs, embeddings)

# 指定本地保存路径
local_faiss_path = "./faiss_index_store"
# 将向量存储保存到本地磁盘，便于后续加载使用
vectorstore.save_local(local_faiss_path)

# 打印保存成功的提示信息
print(f"FAISS index has been saved to {local_faiss_path}")

# 3. 加载索引并执行查询
# 从本地加载 FAISS 索引，需要提供相同的嵌入模型以正确编码查询
# allow_dangerous_deserialization=True 表示允许反序列化（FAISS 使用 pickle，存在安全风险，仅加载可信数据）
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# 定义查询文本
query = "FAISS是做什么的？"
# 在加载的向量存储中执行相似性搜索，返回与查询最相似的 k=1 个文档
results = loaded_vectorstore.similarity_search(query, k=1)

# 打印查询和结果
print(f"\n查询: '{query}'")
print("相似度最高的文档:")
# 遍历结果并打印每个文档的内容
for doc in results:
    print(f"- {doc.page_content}")