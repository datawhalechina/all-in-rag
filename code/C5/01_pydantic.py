from dotenv import load_dotenv
# 从 .env 文件加载环境变量（例如 DEEPSEEK_API_KEY）
load_dotenv()

from typing import List
import os
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_deepseek import ChatDeepSeek

# 初始化 LLM
# 注意：这里没有设 temperature，使用默认值（通常 0.7 左右）；
# 做结构化抽取时一般建议设 temperature=0，输出更稳定、更易解析
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 1. 定义数据结构
# 用 Pydantic 模型描述期望的输出结构：
# - 字段名 + 类型决定 JSON schema；
# - Field 的 description 会被写进 format_instructions，
#   相当于给 LLM 的字段级说明，影响它怎么填值。
class PersonInfo(BaseModel):
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    skills: List[str] = Field(description="技能列表")

# 2. 创建解析器
# PydanticOutputParser 做两件事：
#   a) get_format_instructions() 生成一段「请按此 JSON 格式输出」的提示词；
#   b) parse() 把 LLM 返回的文本反序列化并校验成 PersonInfo 对象，
#      字段缺失/类型不符会抛 OutputParserException。
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# 3. 创建提示模板
# - input_variables=["text"]：运行时需要传入的变量；
# - partial_variables：在构造模板时就固定下来的变量，
#   这里把 format_instructions 一次性填好，后续 invoke 只需传 text。
prompt = PromptTemplate(
    template="请根据以下文本提取信息。\n{format_instructions}\n{text}\n",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# # 打印格式指令
# print("\n--- Format Instructions ---")
# print(parser.get_format_instructions())
# print("--------------------------\n")

# 4. 创建处理链
# LCEL 管道：PromptTemplate → LLM → PydanticOutputParser
# 数据流：dict{"text": ...} → PromptValue → AIMessage → PersonInfo 对象
chain = prompt | llm | parser

# 5. 定义输入文本并执行调用链
text = "张三今年30岁，他擅长Python和Go语言。"
result = chain.invoke({"text": text})

# 6. 打印结果
print("\n--- 解析结果 ---")
# result 已经是 PersonInfo 实例，不是字符串或 dict
print(f"结果类型: {type(result)}")
print(result)
print("--------------------\n")

# 可以直接用属性访问，说明 Pydantic 校验已通过
print(f"姓名: {result.name}")
print(f"年龄: {result.age}")
print(f"技能: {result.skills}")