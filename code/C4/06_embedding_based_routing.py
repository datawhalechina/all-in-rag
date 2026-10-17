from dotenv import load_dotenv
# 从 .env 文件加载环境变量（例如 DEEPSEEK_API_KEY）
load_dotenv()

import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
# 注意：RunnablePassthrough 被重复导入且在本脚本中实际未使用
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnablePassthrough
from langchain_community.utils.math import cosine_similarity
import numpy as np

# 1. 定义路由描述
# 这两段文本不是真正的 Prompt，而是用于「语义路由」的描述句：
# 把用户问题嵌入后，与这两段描述的嵌入向量比较余弦相似度，谁更相似就走哪条链。
sichuan_route_prompt = "你是一位处理川菜的专家。用户的问题是关于麻辣、辛香、重口味的菜肴，例如水煮鱼、麻婆豆腐、鱼香肉丝、宫保鸡丁、花椒、海椒等。"
cantonese_route_prompt = "你是一位处理粤菜的专家。用户的问题是关于清淡、鲜美、原汁原味的菜肴，例如白切鸡、老火靓汤、虾饺、云吞面等。"

# 路由描述列表与对应的路由名称（顺序必须一一对应）
route_prompts = [sichuan_route_prompt, cantonese_route_prompt]
route_names = ["川菜", "粤菜"]

# 初始化嵌入模型，并对路由描述进行向量化
# embed_documents 用于批量嵌入多段文本，返回 List[List[float]]
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
route_prompt_embeddings = embeddings.embed_documents(route_prompts)
print(f"已定义 {len(route_names)} 个路由: {', '.join(route_names)}")

# 2. 定义不同路由的目标链
# 初始化 DeepSeek LLM，temperature=0 让输出更稳定、可复现
llm = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0, 
    api_key=os.getenv("DEEPSEEK_API_KEY")
    )

# 定义川菜和粤菜处理链
# 每条链都是：PromptTemplate -> LLM -> 字符串输出解析器
# 模板中的 {query} 会在 invoke 时由传入字典的 "query" 字段填充
sichuan_chain = (
    PromptTemplate.from_template("你是一位川菜大厨。请用正宗的川菜做法，回答关于「{query}」的问题。")
    | llm
    | StrOutputParser()
)
cantonese_chain = (
    PromptTemplate.from_template("你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{query}」的问题。")
    | llm
    | StrOutputParser()
)

# 路由名称 -> 处理链 的映射表，供 route 函数按名称取链
route_map = { "川菜": sichuan_chain, "粤菜": cantonese_chain }
print("川菜和粤菜的处理链创建成功。\n")

# 3. 创建路由函数
# 输入 info 是一个字典，例如 {"query": "水煮鱼怎么做才嫩？"}
# 函数内部完成：查询嵌入 -> 与各路由描述算余弦相似度 -> 取最高分路由 -> 调用对应链
def route(info):
    # 对用户查询进行嵌入（单条文本，返回一维向量）
    query_embedding = embeddings.embed_query(info["query"])
    
    # 计算与各路由提示的余弦相似度
    # cosine_similarity 期望输入为二维数组，因此用 [query_embedding] 包一层
    # [0] 取出第一行（即与每个路由描述的相似度分数数组）
    similarity_scores = cosine_similarity([query_embedding], route_prompt_embeddings)[0]
    
    # 找到最相似的路由（argmax 返回最大值的索引）
    chosen_route_index = np.argmax(similarity_scores)
    chosen_route_name = route_names[chosen_route_index]
    
    print(f"路由决策: 检测到问题与“{chosen_route_name}”最相似。")
    
    # 获取对应的处理链
    chosen_chain = route_map[chosen_route_name]
    
    # 直接调用选中的链并返回结果
    # 注意：这里传入的 info 仍是 {"query": ...}，正好匹配链模板中的 {query} 占位符
    return chosen_chain.invoke(info)

# 创建完整的路由链
# 用 RunnableLambda 把普通 Python 函数包装成 Runnable，使其可以 .invoke()
full_chain = RunnableLambda(route)


# 4. 运行演示查询
# 前两条分别明显偏向川菜/粤菜，第三条用于验证川菜路由的稳定性
demo_queries = [
    "水煮鱼怎么做才嫩？",        # 应该路由到川菜
    "如何做一碗清淡的云吞面？",    # 应该路由到粤菜
    "麻婆豆腐的核心调料是什么？",  # 应该路由到川菜
]

for i, query in enumerate(demo_queries, 1):
    print(f"\n--- 问题 {i}: {query} ---")
    try:
        # 传入字典，full_chain 会直接返回最终答案（字符串）
        result = full_chain.invoke({"query": query})
        print(f"回答: {result}")
    except Exception as e:
        # 捕获并打印任何执行异常（如 API 调用失败、网络错误等）
        print(f"执行错误: {e}")