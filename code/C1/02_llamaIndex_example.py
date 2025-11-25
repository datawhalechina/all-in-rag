import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
# from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import ZhipuAI

load_dotenv()

# Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.llm = ZhipuAI(model="glm-4-flash", api_key=os.getenv("ZHIPU_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine()

print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?"))