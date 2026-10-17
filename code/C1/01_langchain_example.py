import os

# hugging face镜像设置，如果国内环境无法使用启用该设置
# 将 Hugging Face 的下载端点指向国内镜像，加速模型下载
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量（例如 DEEPSEEK_API_KEY）
load_dotenv()

# 指定要加载的本地 Markdown 文件路径
markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
# 使用 UnstructuredMarkdownLoader 将 Markdown 文件解析为 Document 对象列表
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
# 使用 RecursiveCharacterTextSplitter 将长文档切分成较小的块，便于嵌入和检索
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
# 使用 HuggingFace 的 BAAI/bge-small-zh-v1.5 中文嵌入模型
# device='cpu' 表示在 CPU 上运行；normalize_embeddings=True 表示对向量做归一化
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
# 创建内存向量存储，并将之前切分好的文档块加入其中
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
# 定义 RAG 的提示词模板：要求模型仅根据上下文回答，若信息不足则明确告知
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
)

# 配置大语言模型

# 使用 DeepSeek
# 通过 ChatOpenAI 接口调用 DeepSeek 的 chat 模型
# api_key 从环境变量 DEEPSEEK_API_KEY 读取，base_url 指向 DeepSeek 的 API 地址
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
# 使用相似度搜索，返回与问题最相关的 k=3 个文档块
retrieved_docs = vectorstore.similarity_search(question, k=3)

# 将检索到的文档内容拼接成一个上下文字符串，用两个换行分隔
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 调用大模型：将问题和上下文填入提示词模板，并获取回答
# answer 是一个 AIMessage 对象，包含模型返回的内容
answer = llm.invoke(prompt.format(question=question, context=docs_content))

# 打印模型回答
# AIMessage 对象里除了正文还有 token 用量等元数据，answer.content 才是纯回答文本
print(answer.content)