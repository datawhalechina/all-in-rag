"""
SimpleText2SQLAgent：Text2SQL 代理。

该类整合了知识库检索、SQL 生成、SQL 执行与自动重试修复，
对外提供统一的自然语言查询接口。

主要流程：
1. 连接 SQLite 数据库；
2. 加载知识库（DDL、Q->SQL 示例、表描述等）；
3. 接收用户自然语言问题，检索相关知识；
4. 调用大模型生成 SQL；
5. 执行 SQL，若失败则尝试修复并重试；
6. 返回结构化结果。
"""

import sqlite3
import os
from typing import Dict, Any, List, Tuple
from .knowledge_base import SimpleKnowledgeBase
from .sql_generator import SimpleSQLGenerator


class SimpleText2SQLAgent:
    """Text2SQL代理"""

    def __init__(self, milvus_uri: str = "http://localhost:19530", api_key: str = None):
        """初始化代理。

        参数:
            milvus_uri: Milvus 向量数据库地址，用于知识库检索。
            api_key: DeepSeek API 密钥，用于调用大模型生成 SQL。
        """
        # 初始化知识库（基于 Milvus）
        self.knowledge_base = SimpleKnowledgeBase(milvus_uri)
        # 初始化 SQL 生成器（基于 DeepSeek）
        self.sql_generator = SimpleSQLGenerator(api_key)
        # 数据库连接相关属性
        self.db_path = None
        self.connection = None

        # 配置参数
        self.max_retry_count = 3        # SQL 执行失败后的最大重试次数
        self.top_k_retrieval = 5        # 知识库检索返回的条数
        self.max_result_rows = 100      # 查询结果最大返回行数（自动加 LIMIT）

    def connect_database(self, db_path: str) -> bool:
        """连接 SQLite 数据库。

        参数:
            db_path: SQLite 数据库文件路径。

        返回:
            bool: 连接成功返回 True，失败返回 False。
        """
        try:
            self.db_path = db_path
            self.connection = sqlite3.connect(db_path)
            print(f"成功连接到数据库: {db_path}")
            return True
        except Exception as e:
            print(f"数据库连接失败: {str(e)}")
            return False

    def load_knowledge_base(self):
        """加载知识库数据。

        调用知识库的 load_data 方法，从本地 JSON 文件加载 DDL、
        Q->SQL 示例、表描述等信息，并写入 Milvus。
        """
        self.knowledge_base.load_data()

    def query(self, user_question: str) -> Dict[str, Any]:
        """执行 Text2SQL 查询。

        参数:
            user_question: 用户自然语言问题。

        返回:
            字典，包含以下字段：
                success: 是否成功
                error: 错误信息（成功时为 None）
                sql: 最终执行的 SQL
                results: 查询结果（成功时）
                retry_count: 实际重试次数
        """
        # 如果数据库未连接，直接返回错误
        if not self.connection:
            return {
                "success": False,
                "error": "数据库未连接",
                "sql": None,
                "results": None
            }

        print(f"\n=== 处理查询: {user_question} ===")

        # 1. 从知识库检索相关信息
        print("检索知识库...")
        knowledge_results = self.knowledge_base.search(user_question, self.top_k_retrieval)
        print(f"检索到 {len(knowledge_results)} 条相关信息")

        # 2. 调用大模型生成 SQL
        print("生成SQL...")
        sql = self.sql_generator.generate_sql(user_question, knowledge_results)
        print(f"生成的SQL: {sql}")

        # 3. 执行 SQL（带重试机制）
        retry_count = 0
        while retry_count < self.max_retry_count:
            print(f"执行SQL (尝试 {retry_count + 1}/{self.max_retry_count})...")

            success, result = self._execute_sql(sql)

            if success:
                print("SQL执行成功!")
                return {
                    "success": True,
                    "error": None,
                    "sql": sql,
                    "results": result,
                    "retry_count": retry_count
                }
            else:
                print(f"SQL执行失败: {result}")

                # 如果还有重试机会，则调用大模型修复 SQL
                if retry_count < self.max_retry_count - 1:
                    print("尝试修复SQL...")
                    sql = self.sql_generator.fix_sql(sql, result, knowledge_results)
                    print(f"修复后的SQL: {sql}")

                retry_count += 1

        # 超过最大重试次数仍失败
        return {
            "success": False,
            "error": f"超过最大重试次数 ({self.max_retry_count})",
            "sql": sql,
            "results": None,
            "retry_count": retry_count
        }

    def _execute_sql(self, sql: str) -> Tuple[bool, Any]:
        """执行 SQL 语句。

        参数:
            sql: 待执行的 SQL 语句。

        返回:
            tuple: (是否成功, 结果或错误信息)
                成功时：
                    - 如果是 SELECT，返回 (True, {"columns": ..., "rows": ..., "count": ...})
                    - 如果是非查询语句，返回 (True, "SQL执行成功")
                失败时：
                    - 返回 (False, 错误信息字符串)
        """
        try:
            cursor = self.connection.cursor()

            # 对 SELECT 语句自动添加 LIMIT，防止返回过多数据
            if sql.strip().upper().startswith('SELECT') and 'LIMIT' not in sql.upper():
                sql = f"{sql.rstrip(';')} LIMIT {self.max_result_rows}"

            cursor.execute(sql)

            if sql.strip().upper().startswith('SELECT'):
                # 查询语句：获取列名和所有行，转换为字典列表
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    result_row = {}
                    for i, value in enumerate(row):
                        result_row[columns[i]] = value
                    results.append(result_row)

                cursor.close()
                return True, {
                    "columns": columns,
                    "rows": results,
                    "count": len(results)
                }
            else:
                # 非查询语句：提交事务
                self.connection.commit()
                cursor.close()
                return True, "SQL执行成功"

        except Exception as e:
            # 捕获异常并返回错误信息
            return False, str(e)

    def add_example(self, question: str, sql: str):
        """添加新的 Q->SQL 示例到知识库数据文件中。

        参数:
            question: 自然语言问题。
            sql: 对应的 SQL 语句。
        """
        # 简化版本：直接保存到 JSON 文件，不实时更新 Milvus
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        qsql_path = os.path.join(data_dir, "qsql_examples.json")

        try:
            import json

            # 读取现有数据
            if os.path.exists(qsql_path):
                with open(qsql_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []

            # 添加新示例
            data.append({
                "question": question,
                "sql": sql,
                "database": "sqlite"
            })

            # 写回文件
            with open(qsql_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"已添加新示例: {question}")

        except Exception as e:
            print(f"添加示例失败: {str(e)}")

    def get_table_info(self) -> List[Dict[str, Any]]:
        """获取当前数据库中所有表的结构信息。

        返回:
            列表，每个元素包含：
                table_name: 表名
                columns: 列信息列表，每列包含 name、type、nullable、default、primary_key
        """
        if not self.connection:
            return []

        try:
            cursor = self.connection.cursor()

            # 查询所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            table_info = []
            for table in tables:
                table_name = table[0]

                # 获取表结构
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()

                table_info.append({
                    "table_name": table_name,
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "nullable": not col[3],
                            "default": col[4],
                            "primary_key": bool(col[5])
                        }
                        for col in columns
                    ]
                })

            cursor.close()
            return table_info

        except Exception as e:
            print(f"获取表信息失败: {str(e)}")
            return []

    def cleanup(self):
        """清理资源：关闭数据库连接，删除知识库集合。"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("数据库连接已关闭")

        self.knowledge_base.cleanup()
        print("知识库已清理")