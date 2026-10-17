"""
SimpleKnowledgeBase：基于 Milvus 的 Text2SQL 知识库。

该类负责：
1. 连接 Milvus 向量数据库；
2. 创建/重建知识库集合；
3. 从本地 JSON 文件加载 DDL、问题-SQL 对、表描述等知识；
4. 使用 BGE-M3 生成稠密向量并写入 Milvus；
5. 提供相似度检索接口，供 Text2SQL 代理查询时召回相关知识。
"""

import json
import os
from typing import List, Dict, Any
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 本地 bge-m3 模型路径（仓库根目录 data/models/bge-m3），
# 传本地路径可跳过 HuggingFace 下载（镜像站对 imgs/.DS_Store 返回 403）
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../data/models/bge-m3",
)


class SimpleKnowledgeBase:
    """知识库"""

    def __init__(self, milvus_uri: str = "http://localhost:19530"):
        """
        初始化知识库。

        参数:
            milvus_uri: Milvus 服务地址，默认为本地 19530 端口。
        """
        self.milvus_uri = milvus_uri
        # 创建 Milvus 客户端
        self.client = MilvusClient(uri=milvus_uri)
        # 使用 BGE-M3 混合嵌入模型，这里只使用其稠密向量部分
        self.embedding_function = BGEM3EmbeddingFunction(model_name=MODEL_PATH, use_fp16=False, device="cpu")
        # 知识库集合名称
        self.collection_name = "text2sql_kb"
        # 初始化集合结构
        self._setup_collection()

    def _setup_collection(self):
        """设置集合。如果集合已存在则先删除，再重新创建。"""
        # 如果集合已存在，先删除，保证每次初始化都是干净的结构
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        # 定义集合字段
        fields = [
            # 主键，自动生成，VARCHAR 类型，最大长度 100
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            # 文本内容，例如 DDL、问题-SQL、描述等
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            # 知识类型，可选值：ddl、qsql、description
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=32),
            # 稠密向量字段，维度由嵌入模型决定
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_function.dim["dense"])
        ]

        # 创建集合 schema，并附带描述
        schema = CollectionSchema(fields, description="Text2SQL知识库")

        # 创建集合，一致性级别为 Strong，保证读写立即可见
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            consistency_level="Strong"
        )

        # 准备索引参数
        index_params = self.client.prepare_index_params()
        # 为稠密向量字段添加索引，使用 AUTOINDEX 自动选择索引类型，度量方式为内积 IP
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="IP"
        )

        # 在集合上创建索引
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )

    def load_data(self):
        """加载所有知识库数据。"""
        # 数据目录位于当前文件同级的 data 目录下
        data_dir = os.path.join(os.path.dirname(__file__), "data")

        # 加载 DDL 数据
        ddl_path = os.path.join(data_dir, "ddl_examples.json")
        if os.path.exists(ddl_path):
            with open(ddl_path, 'r', encoding='utf-8') as f:
                ddl_data = json.load(f)
            self._add_ddl_data(ddl_data)

        # 加载“问题 -> SQL”示例数据
        qsql_path = os.path.join(data_dir, "qsql_examples.json")
        if os.path.exists(qsql_path):
            with open(qsql_path, 'r', encoding='utf-8') as f:
                qsql_data = json.load(f)
            self._add_qsql_data(qsql_data)

        # 加载数据库描述数据
        desc_path = os.path.join(data_dir, "db_descriptions.json")
        if os.path.exists(desc_path):
            with open(desc_path, 'r', encoding='utf-8') as f:
                desc_data = json.load(f)
            self._add_description_data(desc_data)

        # 将集合加载到内存，便于后续搜索
        self.client.load_collection(collection_name=self.collection_name)
        print("知识库数据加载完成")

    def _add_ddl_data(self, data: List[Dict]):
        """添加 DDL 数据。

        参数:
            data: 列表，每个元素包含 table_name、ddl_statement、description 等字段。
        """
        contents = []
        types = []

        for item in data:
            # 将一条 DDL 记录拼接为便于向量化的文本
            content = f"表名: {item.get('table_name', '')}\n"
            content += f"DDL: {item.get('ddl_statement', '')}\n"
            content += f"描述: {item.get('description', '')}"

            contents.append(content)
            types.append("ddl")

        # 批量插入
        self._insert_data(contents, types)

    def _add_qsql_data(self, data: List[Dict]):
        """添加“问题 -> SQL”数据。

        参数:
            data: 列表，每个元素包含 question、sql 等字段。
        """
        contents = []
        types = []

        for item in data:
            # 将问题和对应 SQL 拼接为一条知识文本
            content = f"问题: {item.get('question', '')}\n"
            content += f"SQL: {item.get('sql', '')}"

            contents.append(content)
            types.append("qsql")

        # 批量插入
        self._insert_data(contents, types)

    def _add_description_data(self, data: List[Dict]):
        """添加数据库/表描述数据。

        参数:
            data: 列表，每个元素包含 table_name、table_description、columns 等字段。
        """
        contents = []
        types = []

        for item in data:
            # 拼接表名和表描述
            content = f"表名: {item.get('table_name', '')}\n"
            content += f"表描述: {item.get('table_description', '')}\n"

            # 如果包含字段信息，则继续拼接每个字段的名称、描述和类型
            columns = item.get('columns', [])
            if columns:
                content += "字段信息:\n"
                for col in columns:
                    content += f"  - {col.get('name', '')}: {col.get('description', '')} ({col.get('type', '')})\n"

            contents.append(content)
            types.append("description")

        # 批量插入
        self._insert_data(contents, types)

    def _insert_data(self, contents: List[str], types: List[str]):
        """将文本内容向量化后插入 Milvus。

        参数:
            contents: 文本内容列表。
            types: 与 contents 一一对应的知识类型列表。
        """
        if not contents:
            return

        # 使用 BGE-M3 为所有文本生成嵌入向量
        embeddings = self.embedding_function(contents)

        # 构建插入数据，每一行是一个字典，对应 Milvus 中的一条实体
        data_to_insert = []
        for i in range(len(contents)):
            data_to_insert.append({
                "content": contents[i],
                "type": types[i],
                "dense_vector": embeddings["dense"][i]
            })

        # 插入数据到集合
        result = self.client.insert(
            collection_name=self.collection_name,
            data=data_to_insert
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索与查询最相关的知识。

        参数:
            query: 查询文本。
            top_k: 返回的最相似结果数量，默认 5。

        返回:
            列表，每个元素包含 content、type、score。
        """
        # 确保集合已加载到内存
        self.client.load_collection(collection_name=self.collection_name)

        # 将查询文本转换为稠密向量
        query_embeddings = self.embedding_function([query])

        # 在 Milvus 中执行向量相似度搜索
        search_results = self.client.search(
            collection_name=self.collection_name,
            data=query_embeddings["dense"],
            anns_field="dense_vector",
            search_params={"metric_type": "IP"},
            limit=top_k,
            output_fields=["content", "type"]
        )

        # 整理搜索结果
        results = []
        for hit in search_results[0]:
            results.append({
                "content": hit["entity"]["content"],
                "type": hit["entity"]["type"],
                "score": hit["distance"]
            })

        return results

    def cleanup(self):
        """清理资源，删除知识库集合。"""
        try:
            self.client.drop_collection(self.collection_name)
        except:
            # 如果删除失败（例如集合不存在），忽略异常
            pass