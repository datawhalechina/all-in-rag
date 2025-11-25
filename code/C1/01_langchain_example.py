import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from zhipuai import ZhipuAI

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型
# llm = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY")
# )
# 1. 初始化智谱AI客户端（对应原DeepSeek的ChatDeepSeek初始化）
llm_client = ZhipuAI(
    api_key=os.getenv("ZHIPU_API_KEY")  # 环境变量名替换为智谱的API密钥名
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# answer = llm.invoke(prompt.format(question=question, context=docs_content))
# 4. 调用智谱LLM生成答案（替换原llm.invoke）
# --------------------------
# 智谱要求通过「chat.completions.create」调用，且需用messages结构包裹prompt
response = llm_client.chat.completions.create(
    model="GLM-4-Flash",  # 智谱主流模型（可选glm-4/ glm-3-turbo，根据需求选择）
    temperature=0.7,  # 与原DeepSeek的temperature含义一致（控制随机性，0.7保持不变）
    max_tokens=4096,  # 最大生成token数（与原DeepSeek一致，智谱glm-4支持最大8192，可按需调整）
    messages=[  # 智谱强制要求的对话结构：role为user，content为最终prompt
        {"role": "user", "content": prompt.format(question=question, context=docs_content)}
    ]
)

# 智谱返回结果在 response.choices[0].message.content 中
answer = response.choices[0].message.content
print(answer)
