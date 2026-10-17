# -*- coding: utf-8 -*-
# 对比不同 chunk_size / chunk_overlap 对检索和回答的影响
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

docs = UnstructuredMarkdownLoader("../../data/C1/markdown/easy-rl-chapter1.md").load()
print("文档总字符数:", len(docs[0].page_content))

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
llm = ChatOpenAI(model="deepseek-chat", temperature=0.7, max_tokens=1024,
                 api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
prompt = ChatPromptTemplate.from_template(
    "根据上下文回答问题，信息不足就说不知道。\n上下文:\n{context}\n\n问题: {question}\n回答:"
)

# 自己改这组参数做对比：默认是 (4000, 200)
configs = [(4000, 200), (800, 100), (200, 0)]
for cs, co in configs:
    chunks = RecursiveCharacterTextSplitter(chunk_size=cs, chunk_overlap=co).split_documents(docs)
    print(f"\n===== chunk_size={cs}, chunk_overlap={co} | 块数={len(chunks)} | "
          f"各块长度={[len(c.page_content) for c in chunks]} =====")
    vs = InMemoryVectorStore(embeddings)
    vs.add_documents(chunks)
    hits = vs.similarity_search("文中举了哪些例子？", k=3)
    ctx = "\n\n".join(doc.page_content for doc in hits)
    print("--- 检索到的上下文片段开头 ---")
    for i, h in enumerate(hits):
        print(f"  [块{i+1} 长度{len(h.page_content)}] {h.page_content[:100].replace(chr(10), ' ')}...")
    answer = llm.invoke(prompt.format(question="文中举了哪些例子？", context=ctx))
    print("--- 模型回答 ---")
    print(answer.content)
