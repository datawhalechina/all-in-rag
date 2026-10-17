from dotenv import load_dotenv
# 从 .env 文件加载环境变量（例如 DEEPSEEK_API_KEY）
load_dotenv()

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import RunnableBranch

llm = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0, 
    api_key=os.getenv("DEEPSEEK_API_KEY")
    )

# 1. 设置不同菜系的处理链
sichuan_prompt = ChatPromptTemplate.from_template(
    "你是一位川菜大厨。请用正宗的川菜做法，回答关于「{question}」的问题。"
)
sichuan_chain = sichuan_prompt | llm | StrOutputParser()

cantonese_prompt = ChatPromptTemplate.from_template(
    "你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{question}」的问题。"
)
cantonese_chain = cantonese_prompt | llm | StrOutputParser()

# 定义备用通用链
general_prompt = ChatPromptTemplate.from_template(
    "你是一个美食助手。请回答关于「{question}」的问题。"
)
general_chain = general_prompt | llm | StrOutputParser()


# 2. 创建路由链
classifier_prompt = ChatPromptTemplate.from_template(
    """根据用户问题中提到的菜品，将其分类为：['川菜', '粤菜', 或 '其他']。
    不要解释你的理由，只返回一个单词的分类结果。
    问题: {question}"""
)
classifier_chain = classifier_prompt | llm | StrOutputParser()

# 定义路由分支
router_branch = RunnableBranch(
    (lambda x: "川菜" in x["topic"], sichuan_chain),
    (lambda x: "粤菜" in x["topic"], cantonese_chain),
    general_chain  # 默认选项
)

# 组合成完整路由链
full_router_chain = {"topic": classifier_chain, "question": lambda x: x["question"]} | router_branch
print("完整的路由链创建成功。\n")


# 3. 运行演示查询
demo_questions = [
    {"question": "麻婆豆腐怎么做？"},      # 应该路由到川菜
    {"question": "白切鸡的正宗做法是什么？"}, # 应该路由到粤菜
    {"question": "番茄炒蛋需要放糖吗？"}      # 应该路由到其他
]

for i, item in enumerate(demo_questions, 1):
    question = item["question"]
    print(f"\n--- 问题 {i}: {question} ---")
    
    try:
        # 获取路由决策
        topic = classifier_chain.invoke({"question": question})
        print(f"路由决策: {topic}")

        # 执行完整链
        result = full_router_chain.invoke(item)
        print(f"回答: {result}")
    except Exception as e:
        print(f"执行错误: {e}")

