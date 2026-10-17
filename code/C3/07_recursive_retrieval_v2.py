import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 从 .env 文件加载环境变量（例如 DEEPSEEK_API_KEY）
load_dotenv()

# ============================================================
# 配置全局模型（LLM 和 Embedding）
# ============================================================
# 使用 DeepSeek 的 chat 模型作为默认 LLM
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))

# 使用中文 BGE 小模型作为嵌入模型，适合中文文本的向量化
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# ============================================================
# 1. 加载和预处理数据
# ============================================================
# Excel 文件路径（包含多个按年份命名的工作表，如 "年份_1994"）
# 以脚本所在目录为基准，避免依赖运行时的当前工作目录
excel_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../data/C3/excel/movie.xlsx',
)

# 读取 Excel 中所有工作表名称
xls = pd.ExcelFile(excel_file)

# 用于存储两类文档：
#   summary_docs：摘要文档，用于第一步"路由"（定位到具体年份）
#   content_docs：内容文档，用于第二步"问答"（提供实际数据给 LLM）
summary_docs = []
content_docs = []

print("开始加载和处理Excel文件...")

# 遍历每个工作表
for sheet_name in xls.sheet_names:
    # 读取当前工作表为 DataFrame
    df = pd.read_excel(xls, sheet_name=sheet_name)
    
    # -------- 数据清洗 --------
    # 如果存在 "评分人数" 列，需要清洗其格式
    # 原始数据可能形如 "12345人评价"，需要去掉 "人评价" 并转为整数
    if '评分人数' in df.columns:
        df['评分人数'] = df['评分人数'].astype(str).str.replace('人评价', '').str.strip()
        # errors='coerce'：无法转换的值会变成 NaN
        # fillna(0)：把 NaN 填充为 0
        # astype(int)：最终转为整数
        df['评分人数'] = pd.to_numeric(df['评分人数'], errors='coerce').fillna(0).astype(int)

    # -------- 创建摘要文档（用于路由）--------
    # 从工作表名称中提取年份（假设格式为 "年份_1994"）
    year = sheet_name.replace('年份_', '')
    
    # 摘要文本描述该工作表包含的内容，用于向量匹配时快速定位年份
    summary_text = f"这个表格包含了年份为 {year} 的电影信息，包括电影名称、导演、评分、评分人数等。"
    
    # 创建 Document 对象，metadata 中保存 sheet_name，方便后续过滤
    summary_doc = Document(
        text=summary_text,
        metadata={"sheet_name": sheet_name}
    )
    summary_docs.append(summary_doc)
    
    # -------- 创建内容文档（用于最终问答）--------
    # 将 DataFrame 转成字符串形式（保留表格结构，方便 LLM 阅读理解）
    # index=False 表示不显示行号，让文本更干净
    content_text = df.to_string(index=False)
    
    # 同样保存 sheet_name，用于第二步按年份过滤
    content_doc = Document(
        text=content_text,
        metadata={"sheet_name": sheet_name}
    )
    content_docs.append(content_doc)

print("数据加载和处理完成。\n")

# ============================================================
# 2. 构建向量索引
# ============================================================
# 说明：使用默认的内存 SimpleVectorStore，它支持元数据过滤（MetadataFilters）

# 2.1 为摘要文档创建向量索引（用于第一步路由）
summary_index = VectorStoreIndex(summary_docs)

# 2.2 为内容文档创建向量索引（用于第二步问答）
content_index = VectorStoreIndex(content_docs)

print("摘要索引和内容索引构建完成。\n")

# ============================================================
# 3. 定义两步式查询逻辑（路由 + 检索）
# ============================================================
def query_safe_recursive(query_str):
    """
    两步式查询：
      第一步：在摘要索引中找到最相关的年份（路由）
      第二步：在内容索引中，用 metadata 过滤出该年份的内容，再交给 LLM 回答
    """
    print(f"--- 开始执行查询 ---")
    print(f"查询: {query_str}")
    
    # -------- 第一步：路由 --------
    # 在摘要索引中进行向量检索，找到最相关的 1 个摘要文档
    print("\n第一步：在摘要索引中进行路由...")
    summary_retriever = VectorIndexRetriever(index=summary_index, similarity_top_k=1)
    retrieved_nodes = summary_retriever.retrieve(query_str)
    
    # 如果没检索到任何结果，直接返回提示
    if not retrieved_nodes:
        return "抱歉，未能找到相关的电影年份信息。"
    
    # 获取匹配到的工作表名称（作为后续内容过滤的条件）
    matched_sheet_name = retrieved_nodes[0].node.metadata['sheet_name']
    print(f"路由结果：匹配到工作表 -> {matched_sheet_name}")
    
    # -------- 第二步：检索 --------
    # 在内容索引中进行检索，但用 MetadataFilters 限定只搜索 matched_sheet_name 对应的内容
    # ExactMatchFilter 是精确匹配过滤器：metadata 中 sheet_name 必须完全等于路由结果
    # similarity_top_k=1：由于每个年份只有一份内容文档，返回 1 个即可
    print("\n第二步：在内容索引中检索具体信息...")
    content_retriever = VectorIndexRetriever(
        index=content_index,
        similarity_top_k=1,
        filters=MetadataFilters(
            filters=[ExactMatchFilter(key="sheet_name", value=matched_sheet_name)]
        )
    )
    
    # 用检索到的内容节点构造查询引擎，执行最终的问答
    query_engine = RetrieverQueryEngine.from_args(content_retriever)
    response = query_engine.query(query_str)
    
    print("--- 查询执行结束 ---\n")
    return response

# ============================================================
# 4. 执行查询
# ============================================================
# 用户问题：询问 1994 年评分人数最少的电影
query = "1994年评分人数最少的电影是哪一部？"
response = query_safe_recursive(query)

print(f"最终回答: {response}")