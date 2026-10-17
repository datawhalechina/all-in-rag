import os
from langchain_deepseek import ChatDeepSeek 
from langchain_community.document_loaders import BiliBiliLoader
from langchain.chains.query_constructor.base import AttributeInfo
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

from dotenv import load_dotenv
# 从 .env 文件加载环境变量（例如 DEEPSEEK_API_KEY）
load_dotenv()

# 配置日志级别为 INFO，方便观察程序运行过程
logging.basicConfig(level=logging.INFO)

# 1. 初始化视频数据
# 定义待加载的 B 站视频 URL 列表（每个 URL 对应一个 BV 号）
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU", 
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

# 用于存放加载后文档的列表
bili = []
try:
    # 使用 BiliBiliLoader 批量加载视频信息（标题、作者、播放量、时长等）
    loader = BiliBiliLoader(video_urls=video_urls)
    docs = loader.load()
    
    # 遍历每个文档，将原始元数据精简为我们关心的字段
    for doc in docs:
        original = doc.metadata
        
        # 提取基本元数据字段：
        # - title: 视频标题
        # - author: 作者名（原始数据在 owner.name 中）
        # - source: 视频 BV 号（作为唯一标识）
        # - view_count: 播放量（原始数据在 stat.view 中）
        # - length: 视频时长，单位秒（原始数据在 duration 中）
        metadata = {
            'title': original.get('title', '未知标题'),
            'author': original.get('owner', {}).get('name', '未知作者'),
            'source': original.get('bvid', '未知ID'),
            'view_count': original.get('stat', {}).get('view', 0),
            'length': original.get('duration', 0),
        }
        
        # 用精简后的元数据替换原始元数据
        doc.metadata = metadata
        bili.append(doc)
        
except Exception as e:
    # 加载失败时打印错误信息（如网络问题、BV 号无效等）
    print(f"加载BiliBili视频失败: {str(e)}")

# 如果没有任何视频加载成功，则直接退出程序
if not bili:
    print("没有成功加载任何视频，程序退出")
    exit()

# 2. 创建向量存储
# 使用 BAAI/bge-small-zh-v1.5 中文嵌入模型将文档向量化
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
# 将文档存入 Chroma 向量数据库，便于后续相似度检索
vectorstore = Chroma.from_documents(bili, embed_model)

# 3. 配置元数据字段信息
# 这些信息用于告诉查询构造器每个元数据字段的含义和类型
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

# 4. 初始化LLM客户端
# 使用 OpenAI SDK 兼容方式调用 DeepSeek API
# API Key 从环境变量 DEEPSEEK_API_KEY 读取
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 5. 获取所有文档用于排序
# 这里用一个空查询检索全部文档（k=视频数量），目的是拿到完整文档列表
# 注意：这只是为了演示排序，实际场景中应结合检索与排序
all_documents = vectorstore.similarity_search("", k=len(bili)) 

# 6. 执行查询示例
# 定义两个自然语言查询，分别对应按时长升序和按播放量降序
queries = [
    "时间最短的视频",
    "播放量最高的视频"
]

for query in queries:
    print(f"\n--- 原始查询: '{query}' ---")

    # 使用大模型将自然语言转换为排序指令
    # 通过提示词约束模型输出 JSON，包含 sort_by 和 order 两个字段
    prompt = f"""你是一个智能助手，请将用户的问题转换成一个用于排序视频的JSON指令。

你需要识别用户想要排序的字段和排序方向。
- 排序字段必须是 'view_count' (观看次数) 或 'length' (时长) 之一。
- 排序方向必须是 'asc' (升序) 或 'desc' (降序) 之一。

例如:
- '时间最短的视频' 或 '哪个视频时间最短' 应转换为 {{"sort_by": "length", "order": "asc"}}
- '播放量最高的视频' 或 '哪个视频最火' 应转换为 {{"sort_by": "view_count", "order": "desc"}}

请根据以下问题生成JSON指令:
原始问题: "{query}"

JSON指令:"""
    
    # 调用 DeepSeek 模型，temperature=0 保证输出稳定
    # response_format 指定为 json_object，强制模型返回 JSON 格式
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    try:
        import json
        # 取出模型返回的 JSON 字符串并解析为字典
        instruction_str = response.choices[0].message.content
        instruction = json.loads(instruction_str)
        print(f"--- 生成的排序指令: {instruction} ---")

        # 从指令中提取排序字段和排序方向
        sort_by = instruction.get('sort_by')
        order = instruction.get('order')

        # 校验指令合法性：字段只能是 length/view_count，方向只能是 asc/desc
        if sort_by in ['length', 'view_count'] and order in ['asc', 'desc']:
            # 在代码中执行排序
            # reverse_order 为 True 表示降序，False 表示升序
            reverse_order = (order == 'desc')
            # 按指定元数据字段对文档排序，缺失字段默认按 0 处理
            sorted_docs = sorted(all_documents, key=lambda doc: doc.metadata.get(sort_by, 0), reverse=reverse_order)
            
            # 获取排序后的第一个结果（即最符合查询条件的视频）
            if sorted_docs:
                doc = sorted_docs[0]
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
                print("没有找到任何视频")
        else:
            print("生成的指令无效，无法执行排序")

    except (json.JSONDecodeError, KeyError) as e:
        # 处理 JSON 解析失败或字段缺失的情况
        print(f"解析或执行指令失败: {e}")