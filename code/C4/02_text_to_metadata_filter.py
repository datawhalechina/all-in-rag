import os

from dotenv import load_dotenv
# 从 .env 文件加载环境变量（例如 DEEPSEEK_API_KEY）
load_dotenv()

# 使用国内 HuggingFace 镜像（必须在导入模型相关库之前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_deepseek import ChatDeepSeek 
from langchain_community.document_loaders import BiliBiliLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# 配置日志，方便查看 SelfQueryRetriever 的详细执行过程
logging.basicConfig(level=logging.INFO)

# ============================================================
# 1. 初始化视频数据
# ============================================================
# 待加载的 Bilibili 视频链接列表
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU", 
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

# 用于存放加载后的文档（Document）对象
bili = []

try:
    # 使用 BiliBiliLoader 加载视频信息（标题、作者、播放量、时长等）
    loader = BiliBiliLoader(video_urls=video_urls)
    docs = loader.load()
    
    # 遍历每个文档，重新整理元数据（只保留后续检索需要的字段）
    for doc in docs:
        original = doc.metadata  # 原始元数据（结构较复杂）
        
        # 提取并构建简洁的元数据字段
        metadata = {
            'title': original.get('title', '未知标题'),                     # 视频标题
            'author': original.get('owner', {}).get('name', '未知作者'),    # 作者名
            'source': original.get('bvid', '未知ID'),                       # 视频 BV 号
            'view_count': original.get('stat', {}).get('view', 0),          # 播放量
            'length': original.get('duration', 0),                          # 视频时长（秒）
        }
        
        # 用精简后的元数据替换原始元数据
        doc.metadata = metadata
        bili.append(doc)
        
except Exception as e:
    print(f"加载BiliBili视频失败: {str(e)}")

# 如果没有任何视频加载成功，则退出程序
if not bili:
    print("没有成功加载任何视频，程序退出")
    exit()

# ============================================================
# 2. 创建向量存储
# ============================================================
# 使用 BGE 中文小模型作为嵌入模型，将文档内容向量化
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 基于加载的文档创建 Chroma 向量数据库
vectorstore = Chroma.from_documents(bili, embed_model)

# ============================================================
# 3. 配置元数据字段信息
# ============================================================
# 告诉 SelfQueryRetriever 每个元数据字段的名称、描述和类型
# 这样 LLM 才能正确地把自然语言查询转换成结构化过滤器
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="视频标题（字符串）",
        type="string", 
    ),
    AttributeInfo(
        name="author",
        description="视频作者（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="视频观看次数（整数）",
        type="integer",
    ),
    AttributeInfo(
        name="length",
        description="视频长度（整数）",
        type="integer"
    )
]

# ============================================================
# 4. 创建自查询检索器（SelfQueryRetriever）
# ============================================================
# 初始化 DeepSeek 聊天模型，temperature=0 保证输出稳定
llm = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0, 
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 创建 SelfQueryRetriever：
# - 它会先用 LLM 解析用户查询，提取出「语义查询」和「元数据过滤器」
# - 然后在向量库中执行带过滤条件的相似度检索
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="记录视频标题、作者、观看次数等信息的视频元数据",  # 文档内容描述
    metadata_field_info=metadata_field_info,   # 元数据字段定义
    enable_limit=True,                         # 允许 LLM 生成 limit（如“前3个”）
    verbose=True                               # 打印解析过程，便于调试
)

# ============================================================
# 5. 执行查询示例
# ============================================================
# 测试两个自然语言查询：
#   - “时间最短的视频” → 应该会解析为按 length 升序，取 1 条
#   - “时长大于600秒的视频” → 应该会解析为 length > 600 的过滤条件
queries = [
    "时间最短的视频",
    "时长大于600秒的视频"
]

for query in queries:
    print(f"\n--- 查询: '{query}' ---")
    # 调用检索器，内部会自动进行查询解析和向量检索
    results = retriever.invoke(query)
    
    if results:
        for doc in results:
            # 从元数据中取出各字段并展示
            title = doc.metadata.get('title', '未知标题')
            author = doc.metadata.get('author', '未知作者')
            view_count = doc.metadata.get('view_count', '未知')
            length = doc.metadata.get('length', '未知')
            print(f"标题: {title}")
            print(f"作者: {author}")
            print(f"观看次数: {view_count}")
            print(f"时长: {length}秒")
            print("="*50)
    else:
        print("未找到匹配的视频")