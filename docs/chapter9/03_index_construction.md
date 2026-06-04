# 第三节 Milvus索引构建

在图RAG系统中，索引构建是连接图数据和向量检索的关键环节。本节介绍如何将图数据转换为可检索的向量索引。

在第三章中，我们已经详细介绍了Milvus的基本概念、部署方式和基础操作。本节将在此基础上，专门针对图RAG场景进行深度应用。如果你对Milvus还不熟悉，建议先阅读[Milvus介绍及多模态检索实践](https://github.com/datawhalechina/all-in-rag/blob/main/docs/chapter3/09_milvus.md)。

> [本节完整代码](https://github.com/datawhalechina/all-in-rag/blob/main/code/C9/rag_modules/milvus_index_construction.py)

## 一、索引构建概述

### 1.1 索引构建流程

图RAG的索引构建需要将从图数据库构建的结构化文档转换为向量表示，并存储到向量数据库中：

```mermaid
flowchart LR
    A[图数据库] --> B[文档构建]
    B --> C[文档分块]
    C --> D[向量化]
    D --> E[Milvus索引]
    
    style A fill:#e1f5fe
    style E fill:#e8f5e8
```

### 1.2 核心组件

- **文档构建器**：从图数据构建结构化文档
- **分块处理器**：智能分块策略
- **向量化模型**：文本转向量
- **Milvus索引**：高性能向量存储和检索

## 二、Milvus索引构建实现

### 2.1 索引构建器核心架构

```python
class MilvusIndexConstructionModule:
    """Milvus索引构建模块 - 负责向量化和Milvus索引构建"""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 19530,
                 collection_name: str = "cooking_knowledge",
                 dimension: int = 512,
                 model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.model_name = model_name

        self.client = None
        self.embeddings = None
        self.collection_created = False

        self._setup_client()
        self._setup_embeddings()
```

**代码解读**：
- **模块化设计**：将Milvus操作封装为独立模块，便于复用和维护
- **配置灵活性**：支持自定义Milvus连接参数和嵌入模型
- **中文优化**：默认使用`BAAI/bge-small-zh-v1.5`，专门针对中文文本优化
- **延迟初始化**：在构造函数中设置连接，避免启动时的阻塞

### 2.2 向量化处理

```python
def _vectorize_documents(self, documents: List[Document]) -> Tuple[List[List[float]], List[Dict]]:
    """文档向量化处理"""
    vectors = []
    metadatas = []
    
    for i, doc in enumerate(documents):
        try:
            # 向量化文档内容
            vector = self.embedding_model.embed_query(doc.page_content)
            vectors.append(vector)
            
            # 准备元数据
            metadata = {
                "id": i,
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "parent_id": doc.metadata.get("parent_id", ""),
                # ... 其他元数据
            }
            metadatas.append(metadata)
            
        except Exception as e:
            logger.error(f"文档 {i} 向量化失败: {e}")
            continue
    
    return vectors, metadatas
```

### 2.3 图RAG专用集合Schema设计

```python
def _create_collection_schema(self):
    """创建集合schema"""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=150, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=15000),
        FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="recipe_name", dtype=DataType.VARCHAR, max_length=300),
        FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="cuisine_type", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="difficulty", dtype=DataType.INT64),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100)
    ]

    schema = CollectionSchema(
        fields=fields,
        description="中式烹饪知识图谱向量集合"
    )
    return schema
```

**Schema设计亮点**：
- **图数据特化**：专门为烹饪知识图谱设计的字段结构
- **丰富元数据**：包含菜谱名称、节点类型、菜系、难度等图谱特有信息
- **长度优化**：根据实际数据特点设置合理的字段长度限制
- **检索友好**：所有关键字段都可用于过滤和检索条件

## 三、索引优化策略

### 3.1 批量插入优化

```python
def _batch_insert(self, vectors: List[List[float]], metadatas: List[Dict]):
    """批量插入优化"""
    batch_size = self.config.batch_size
    collection_name = self.config.milvus_collection_name
    
    for i in range(0, len(vectors), batch_size):
        batch_vectors = vectors[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        
        # 准备插入数据
        insert_data = [
            [meta["id"] for meta in batch_metadatas],           # id
            batch_vectors,                                       # vector
            [meta["content"] for meta in batch_metadatas],      # content
            [meta["source"] for meta in batch_metadatas],       # source
            [meta["chunk_id"] for meta in batch_metadatas],     # chunk_id
            [meta["parent_id"] for meta in batch_metadatas],    # parent_id
        ]
        
        # 执行插入
        self.milvus_client.insert(collection_name, insert_data)
        logger.info(f"批次 {i//batch_size + 1} 插入完成，数量: {len(batch_vectors)}")
```

### 3.2 索引创建

```python
def _create_index(self):
    """创建向量索引"""
    collection_name = self.config.milvus_collection_name
    
    # 索引参数
    index_params = {
        "metric_type": "COSINE",    # 余弦相似度
        "index_type": "IVF_FLAT",   # 索引类型
        "params": {"nlist": 1024}   # 索引参数
    }
    
    # 创建索引
    self.milvus_client.create_index(
        collection_name=collection_name,
        field_name="vector",
        index_params=index_params
    )
    
    # 加载集合到内存
    self.milvus_client.load_collection(collection_name)
    
    logger.info("向量索引创建完成")
```

## 四、索引构建流程

### 4.1 核心向量构建流程

```python
def build_vector_index(self, chunks: List[Document]) -> bool:
    """构建向量索引"""
    logger.info(f"正在构建Milvus向量索引，文档数量: {len(chunks)}...")

    try:
        # 1. 创建集合（如果schema不兼容则强制重新创建）
        if not self.create_collection(force_recreate=True):
            return False

        # 2. 准备数据
        logger.info("正在生成向量embeddings...")
        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)

        # 3. 准备插入数据
        entities = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            entity = {
                "id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                "vector": vector,
                "text": self._safe_truncate(chunk.page_content, 15000),
                "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
                "recipe_name": self._safe_truncate(chunk.metadata.get("recipe_name", ""), 300),
                # ... 更多字段
            }
            entities.append(entity)

        # 4. 批量插入数据
        batch_size = 100
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            self.client.insert(collection_name=self.collection_name, data=batch)
```

**关键技术点解读**：

1. **强制重建策略**：`force_recreate=True`确保Schema一致性，避免字段不匹配错误

2. **批量向量化**：一次性处理所有文档的向量化，提高效率
   ```python
   texts = [chunk.page_content for chunk in chunks]
   vectors = self.embeddings.embed_documents(texts)  # 批量处理
   ```

3. **安全截断机制**：`_safe_truncate`方法防止字段长度超限
   ```python
   def _safe_truncate(self, text: str, max_length: int) -> str:
       if text is None:
           return ""
       return str(text)[:max_length]
   ```

4. **图数据元数据保留**：完整保留图谱中的结构化信息，支持后续的复合检索

### 4.2 索引验证

```python
def verify_index(self) -> bool:
    """验证索引构建结果"""
    try:
        collection_name = self.config.milvus_collection_name
        
        # 检查集合状态
        collection_info = self.milvus_client.describe_collection(collection_name)
        logger.info(f"集合信息: {collection_info}")
        
        # 检查数据量
        count = self.milvus_client.query(
            collection_name=collection_name,
            expr="",
            output_fields=["count(*)"]
        )
        logger.info(f"索引中文档数量: {count}")
        
        # 简单检索测试
        test_results = self.milvus_client.search(
            collection_name=collection_name,
            data=[[0.1] * self.config.embedding_dim],  # 测试向量
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=1
        )
        
        logger.info("索引验证通过")
        return True

    except Exception as e:
        logger.error(f"索引验证失败: {e}")
        return False
```

## 五、为什么从FAISS切换到Milvus？

在第八章中，使用的是FAISS作为向量存储方案。虽然FAISS在研究和原型开发中表现出色，但在生产环境和复杂应用场景下，Milvus提供了更多优势：

**FAISS的局限性**：
- **纯库模式**：FAISS是一个向量搜索库，缺乏数据库的完整功能
- **无持久化**：需要手动管理数据持久化和备份
- **单机限制**：难以实现分布式部署和水平扩展
- **元数据支持有限**：无法高效存储和查询复杂的结构化元数据
- **并发性能**：在高并发场景下性能受限

**Milvus的优势**：
- **完整数据库功能**：提供CRUD操作、事务支持、数据一致性保证
- **云原生架构**：支持分布式部署、自动扩缩容、高可用性
- **丰富的元数据支持**：支持复杂Schema设计，适合图RAG的多维度数据
- **生产级特性**：监控、日志、备份恢复等企业级功能
