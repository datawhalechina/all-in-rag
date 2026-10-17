import os
from typing import List, Dict, Any
from langchain_deepseek import ChatDeepSeek
from langchain.schema import HumanMessage, SystemMessage


class SimpleSQLGenerator:
    """简化的SQL生成器。

    使用 DeepSeek 大语言模型，根据用户自然语言问题和知识库检索结果，
    生成对应的 SQL 查询语句；当 SQL 执行出错时，也能根据错误信息进行修复。
    """

    def __init__(self, api_key: str = None):
        """
        初始化 SQL 生成器。

        参数:
            api_key: DeepSeek API 密钥。如果未提供，则从环境变量 DEEPSEEK_API_KEY 读取。
        """
        # 创建 ChatDeepSeek 实例，使用 deepseek-chat 模型，温度设为 0 以保证输出稳定
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0,
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY")
        )

    def generate_sql(self, user_query: str, knowledge_results: List[Dict[str, Any]]) -> str:
        """根据用户问题和知识库检索结果生成 SQL 语句。

        参数:
            user_query: 用户提出的自然语言问题。
            knowledge_results: 知识库检索结果列表，每项包含 content、type 等字段。

        返回:
            生成的 SQL 语句（已清理掉 Markdown 代码块标记）。
        """
        # 将知识库检索结果整理成上下文文本
        context = self._build_context(knowledge_results)

        # 构建提示词，指导模型生成 SQL
        prompt = f"""你是一个SQL专家。请根据以下信息将用户问题转换为SQL查询语句。

数据库信息：
{context}

用户问题：{user_query}

要求：
1. 只返回SQL语句，不要包含任何解释
2. 确保SQL语法正确
3. 使用上下文中提供的表名和字段名
4. 如果需要JOIN，请根据表结构进行合理关联

SQL语句："""

        # 构造消息列表，这里只使用 HumanMessage
        messages = [HumanMessage(content=prompt)]
        # 调用大模型
        response = self.llm.invoke(messages)

        # 清理模型返回的内容，去除可能的 Markdown 代码块标记
        sql = response.content.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]

        return sql.strip()

    def fix_sql(self, original_sql: str, error_message: str, knowledge_results: List[Dict[str, Any]]) -> str:
        """根据错误信息修复 SQL 语句。

        参数:
            original_sql: 原始出错的 SQL 语句。
            error_message: 数据库执行时返回的错误信息。
            knowledge_results: 知识库检索结果列表，用于提供上下文。

        返回:
            修复后的 SQL 语句（已清理掉 Markdown 代码块标记）。
        """
        # 构建上下文
        context = self._build_context(knowledge_results)

        # 构建修复提示词
        prompt = f"""请修复以下SQL语句的错误。

数据库信息：
{context}

原始SQL：
{original_sql}

错误信息：
{error_message}

请返回修复后的SQL语句（只返回SQL，不要解释）："""

        # 构造消息并调用模型
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)

        # 清理返回内容，去除 Markdown 代码块标记
        fixed_sql = response.content.strip()
        if fixed_sql.startswith("```sql"):
            fixed_sql = fixed_sql[6:]
        if fixed_sql.startswith("```"):
            fixed_sql = fixed_sql[3:]
        if fixed_sql.endswith("```"):
            fixed_sql = fixed_sql[:-3]

        return fixed_sql.strip()

    def _build_context(self, knowledge_results: List[Dict[str, Any]]) -> str:
        """将知识库检索结果按类型分组，拼接成上下文字符串。

        参数:
            knowledge_results: 知识库检索结果列表。

        返回:
            格式化后的上下文字符串，包含表结构、表/字段描述和查询示例。
        """
        context = ""

        # 按知识类型分组
        ddl_info = []          # 表结构信息（DDL）
        qsql_examples = []     # 问题-SQL 示例
        descriptions = []      # 表和字段描述

        for result in knowledge_results:
            if result["type"] == "ddl":
                ddl_info.append(result["content"])
            elif result["type"] == "qsql":
                qsql_examples.append(result["content"])
            elif result["type"] == "description":
                descriptions.append(result["content"])

        # 拼接表结构信息
        if ddl_info:
            context += "=== 表结构信息 ===\n"
            context += "\n".join(ddl_info) + "\n\n"

        # 拼接表和字段描述
        if descriptions:
            context += "=== 表和字段描述 ===\n"
            context += "\n".join(descriptions) + "\n\n"

        # 拼接查询示例
        if qsql_examples:
            context += "=== 查询示例 ===\n"
            context += "\n".join(qsql_examples) + "\n\n"

        return context