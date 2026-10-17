import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import IndexNode
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

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
# 1. 加载数据并为每个工作表创建查询引擎和摘要节点
# ============================================================
# Excel 文件路径（以脚本所在目录为基准，避免依赖运行时的当前工作目录）
excel_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../data/C3/excel/movie.xlsx',
)

# 读取 Excel 文件中的所有工作表名称
xls = pd.ExcelFile(excel_file)

# 用于存储：工作表名称 -> 对应的 PandasQueryEngine
df_query_engines = {}

# 用于存储所有摘要节点（IndexNode），后续构建顶层向量索引
all_nodes = []

# 遍历每一个工作表
for sheet_name in xls.sheet_names:
    # 读取当前工作表为 DataFrame
    df = pd.read_excel(xls, sheet_name=sheet_name)
    
    # 为当前 DataFrame 创建一个 PandasQueryEngine
    # 这个引擎能够理解自然语言问题，并生成 pandas 代码来查询该表格
    # verbose=True 会打印出生成的 pandas 代码，便于调试
    query_engine = PandasQueryEngine(df=df, llm=Settings.llm, verbose=True)
    
    # 从工作表名称中提取年份（假设工作表名称格式为 "年份_1994"）
    year = sheet_name.replace('年份_', '')
    
    # 为当前工作表创建一个摘要文本，用于在顶层向量检索时匹配
    summary = f"这个表格包含了年份为 {year} 的电影信息，可以用来回答关于这一年电影的具体问题。"
    
    # 创建一个 IndexNode（索引节点），它只包含摘要文本，并关联到具体查询引擎的 ID
    # index_id 必须与 df_query_engines 中的键一致，以便递归检索器能够路由
    node = IndexNode(text=summary, index_id=sheet_name)
    all_nodes.append(node)
    
    # 存储工作表名称到其查询引擎的映射
    df_query_engines[sheet_name] = query_engine

# ============================================================
# 2. 创建顶层索引（只包含摘要节点）
# ============================================================
# 使用所有摘要节点构建一个向量索引
# 这个索引的作用是：根据用户问题，快速找到最相关的年份（工作表）
vector_index = VectorStoreIndex(all_nodes)

# ============================================================
# 3. 创建递归检索器
# ============================================================
# 3.1 创建顶层检索器：在摘要节点中进行向量相似度检索
# similarity_top_k=1 表示只返回最匹配的一个摘要节点（即一个年份）
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# 3.2 创建递归检索器 RecursiveRetriever
# 它的工作流程：
#   - 首先使用 "vector" 检索器找到最相关的摘要节点
#   - 然后根据该节点的 index_id，从 query_engine_dict 中取出对应的 PandasQueryEngine
#   - 最后将原始问题交给这个 PandasQueryEngine 来生成答案
recursive_retriever = RecursiveRetriever(
    "vector",                                  # 顶层检索器的名称
    retriever_dict={"vector": vector_retriever},  # 检索器字典
    query_engine_dict=df_query_engines,        # 查询引擎字典：index_id -> PandasQueryEngine
    verbose=True,                              # 打印详细的路由过程
)

# ============================================================
# 4. 创建查询引擎
# ============================================================
# 将递归检索器包装成一个标准的查询引擎，这样就可以直接调用 .query()
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

# ============================================================
# 5. 执行查询
# ============================================================
# 用户问题：询问 1994 年评分人数最少的电影
query = "1994年评分人数最少的电影是哪一部？"
print(f"查询: {query}")

# 执行查询，递归检索器会自动路由到 "年份_1994" 对应的 PandasQueryEngine
response = query_engine.query(query)
print(f"回答: {response}")