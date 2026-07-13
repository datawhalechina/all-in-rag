"""
混合检索模块
基于双层检索范式：实体级 + 主题级检索
结合 BM25（jieba 分词）、向量检索与图键值索引，使用 RRF 融合
"""

import json
import logging
import hashlib
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from neo4j import GraphDatabase
from .graph_indexing import GraphIndexingModule

logger = logging.getLogger(__name__)

# 中文停用词表：助词 / 连词 / 疑问词 / 人称 / 语气词 / 动词修饰
# 不引第三方停用词包，按烹饪问答场景手挑（覆盖 testset 高频虚词）
_CHINESE_STOPWORDS = set("""
的 了 和 是 在 我 有 就 不 也 都 还 这 那 一 个 与 及 等 上 下 中 为 以 于 从 把 被 让 使 又 而 但 或
什么 怎么 如何 哪些 哪个 哪里 谁 多少 几 你 他 她 它 我们 他们 她们 它们
请问 请 想 要 需要 能 可以 应该 会 啊 呢 吧 嘛 吗 哦 呀 哈
之 其 此 该 即 各 每 些 种 类 时 后 前 里 外 内 间 已经 正在 一些 一下
""".split())

# RRF 融合的常数 k：Cormack et al. 2009 默认值
_RRF_K = 60

@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str
    node_id: str
    node_type: str
    relevance_score: float
    retrieval_level: str  # 'low' or 'high'
    metadata: Dict[str, Any]

class HybridRetrievalModule:
    """
    混合检索模块
    核心特点：
    1. 双层检索范式（实体级 + 主题级，基于图键值索引）
    2. BM25 关键词检索（jieba 分词 + 停用词过滤）
    3. 向量检索（Milvus）+ 一跳邻居扩展
    4. RRF (Reciprocal Rank Fusion) 融合三路结果
    """

    def __init__(self, config, milvus_module, data_module, llm_client):
        self.config = config
        self.milvus_module = milvus_module
        self.data_module = data_module
        self.llm_client = llm_client
        self.driver = None

        # BM25 索引 + 原始文档（按索引位置对齐）
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus_docs: List[Document] = []

        # 图索引模块
        self.graph_indexing = GraphIndexingModule(config, llm_client)
        self.graph_indexed = False

    def initialize(self, chunks: List[Document]):
        """初始化检索系统"""
        logger.info("初始化混合检索模块...")

        # 连接Neo4j
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )

        # 初始化 BM25（jieba 分词 + 中文停用词过滤）
        if chunks:
            self.bm25_corpus_docs = list(chunks)
            tokenized_corpus = [self._tokenize_chinese(d.page_content) for d in chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            avg_tokens = sum(len(t) for t in tokenized_corpus) / max(1, len(tokenized_corpus))
            logger.info(
                f"BM25(jieba+stopwords) 索引构建完成，文档数: {len(chunks)}，"
                f"平均 token 数: {avg_tokens:.1f}"
            )

        # 初始化图索引
        self._build_graph_index()

        # 初始化父文档映射，每个nodeid对应该chunk所属父文档的document
        self._parent_doc_map = self._build_parent_doc_map()
        logger.info(f"父文档映射构建完成，菜谱文档数: {len(self._parent_doc_map)}")

    @staticmethod
    def _tokenize_chinese(text: str) -> List[str]:
        """jieba 精确分词 + 停用词 / 空白 / 单字符过滤"""
        if not text:
            return []
        tokens = jieba.lcut(text)
        return [
            t for t in tokens
            if t.strip() and t not in _CHINESE_STOPWORDS and not t.isspace()
        ]
        
    def _build_graph_index(self):
        """构建图索引"""
        if self.graph_indexed:
            return
            
        logger.info("开始构建图索引...")
        
        try:
            # 获取图数据
            recipes = self.data_module.recipes
            ingredients = self.data_module.ingredients
            cooking_steps = self.data_module.cooking_steps
            
            # 创建实体键值对
            self.graph_indexing.create_entity_key_values(recipes, ingredients, cooking_steps)
            
            # 创建关系键值对（这里需要从Neo4j获取关系数据）
            relationships = self._extract_relationships_from_graph()
            self.graph_indexing.create_relation_key_values(relationships)
            
            # 去重优化
            self.graph_indexing.deduplicate_entities_and_relations()
            
            self.graph_indexed = True
            stats = self.graph_indexing.get_statistics()
            logger.info(f"图索引构建完成: {stats}")
            
        except Exception as e:
            logger.error(f"构建图索引失败: {e}")
            
    def _extract_relationships_from_graph(self) -> List[Tuple[str, str, str]]:
        """从Neo4j图中提取关系"""
        relationships = []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (source)-[r]->(target)
                WHERE source.nodeId >= '200000000' OR target.nodeId >= '200000000'
                RETURN source.nodeId as source_id, type(r) as relation_type, target.nodeId as target_id
                LIMIT 1000
                """
                result = session.run(query)
                
                for record in result:
                    relationships.append((
                        record["source_id"],
                        record["relation_type"],
                        record["target_id"]
                    ))
                    
        except Exception as e:
            logger.error(f"提取图关系失败: {e}")
            
        return relationships
            
    def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """
        提取查询关键词：实体级 + 主题级
        """
        prompt = f"""
        作为烹饪知识助手，请分析以下查询并提取关键词，分为两个层次：

        查询：{query}

        提取规则：
        1. 实体级关键词：具体的食材、菜品名称、工具、品牌等有形实体
           - 例如：鸡胸肉、西兰花、红烧肉、平底锅、老干妈
           - 对于抽象查询，推测相关的具体食材/菜品

        2. 主题级关键词：抽象概念、烹饪主题、饮食风格、营养特点等
           - 例如：减肥、低热量、川菜、素食、下饭菜、快手菜
           - 排除动作词：推荐、介绍、制作、怎么做等

        示例：
        查询："推荐几个减肥菜" 
        {{
            "entity_keywords": ["鸡胸肉", "西兰花", "水煮蛋", "胡萝卜", "黄瓜"],
            "topic_keywords": ["减肥", "低热量", "高蛋白", "低脂"]
        }}

        查询："川菜有什么特色"
        {{
            "entity_keywords": ["麻婆豆腐", "宫保鸡丁", "水煮鱼", "辣椒", "花椒"],
            "topic_keywords": ["川菜", "麻辣", "香辣", "下饭菜"]
        }}

        请严格按照JSON格式返回，不要包含多余的文字：
        {{
            "entity_keywords": ["实体1", "实体2", ...],
            "topic_keywords": ["主题1", "主题2", ...]
        }}
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            entity_keywords = result.get("entity_keywords", [])
            topic_keywords = result.get("topic_keywords", [])
            
            logger.info(f"关键词提取完成 - 实体级: {entity_keywords}, 主题级: {topic_keywords}")
            return entity_keywords, topic_keywords
            
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            # 降级方案：简单的关键词分割
            keywords = query.split()
            return keywords[:3], keywords[3:6] if len(keywords) > 3 else keywords
    
    def entity_level_retrieval(self, entity_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """
        实体级检索：专注于具体实体和关系
        使用图索引的键值对结构进行检索
        """
        results = []
        
        # 1. 使用图索引进行实体检索
        for keyword in entity_keywords:
            # 检索匹配的实体
            entities = self.graph_indexing.get_entities_by_key(keyword)
            
            for entity in entities:
                # 获取邻居信息
                neighbors = self._get_node_neighbors(entity.metadata["node_id"], max_neighbors=2)
                
                # 构建增强内容
                enhanced_content = entity.value_content
                if neighbors:
                    enhanced_content += f"\n相关信息: {', '.join(neighbors)}"
                
                results.append(RetrievalResult(
                    content=enhanced_content,
                    node_id=entity.metadata["node_id"],
                    node_type=entity.entity_type,
                    relevance_score=0.9,  # 精确匹配得分较高
                    retrieval_level="entity",
                    metadata={
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "index_keys": entity.index_keys,
                        "matched_keyword": keyword
                    }
                ))
        
        # 2. 如果图索引结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_entity_level_search(entity_keywords, top_k - len(results))
            results.extend(neo4j_results)
            
        # 3. 按相关性排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"实体级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]
    
    def _neo4j_entity_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j补充检索"""
        results = []
        
        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                CALL db.index.fulltext.queryNodes('recipe_fulltext_index', keyword + '*') 
                YIELD node, score
                WHERE node:Recipe
                RETURN 
                    node.nodeId as node_id,
                    node.name as name,
                    node.description as description,
                    labels(node) as labels,
                    score
                ORDER BY score DESC
                LIMIT $limit
                """
                
                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })
                
                for record in result:
                    content_parts = []
                    if record["name"]:
                        content_parts.append(f"菜品: {record['name']}")
                    if record["description"]:
                        content_parts.append(f"描述: {record['description']}")
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=record["node_id"],
                        node_type="Recipe",
                        relevance_score=float(record["score"]) * 0.7,  # 补充检索得分较低
                        retrieval_level="entity",
                        metadata={
                            "name": record["name"],
                            "labels": record["labels"],
                            "source": "neo4j_fallback"
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Neo4j补充检索失败: {e}")
            
        return results
    
    def topic_level_retrieval(self, topic_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """
        主题级检索：专注于广泛主题和概念
        使用图索引的关系键值对结构进行主题检索
        """
        results = []
        
        # 1. 使用图索引进行关系/主题检索
        for keyword in topic_keywords:
            # 检索匹配的关系
            relations = self.graph_indexing.get_relations_by_key(keyword)
            
            for relation in relations:
                # 获取相关实体信息
                source_entity = self.graph_indexing.entity_kv_store.get(relation.source_entity)
                target_entity = self.graph_indexing.entity_kv_store.get(relation.target_entity)
                
                if source_entity and target_entity:
                    # 构建丰富的主题内容
                    content_parts = [
                        f"主题: {keyword}",
                        relation.value_content,
                        f"相关菜品: {source_entity.entity_name}",
                        f"相关信息: {target_entity.entity_name}"
                    ]
                    
                    # 添加源实体的详细信息
                    if source_entity.entity_type == "Recipe":
                        newline = '\n'
                        content_parts.append(f"菜品详情: {source_entity.value_content.split(newline)[0]}")
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=relation.source_entity,  # 以主要实体为ID
                        node_type=source_entity.entity_type,
                        relevance_score=0.95,  # 主题匹配得分
                        retrieval_level="topic",
                        metadata={
                            "relation_id": relation.relation_id,
                            "relation_type": relation.relation_type,
                            "source_name": source_entity.entity_name,
                            "target_name": target_entity.entity_name,
                            "matched_keyword": keyword,
                            "index_keys": relation.index_keys
                        }
                    ))
        
        # 2. 使用实体的分类信息进行主题检索
        for keyword in topic_keywords:
            entities = self.graph_indexing.get_entities_by_key(keyword)
            for entity in entities:
                if entity.entity_type == "Recipe":
                    # 构建分类主题内容
                    content_parts = [
                        f"主题分类: {keyword}",
                        entity.value_content
                    ]
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=entity.metadata["node_id"],
                        node_type=entity.entity_type,
                        relevance_score=0.85,  # 分类匹配得分
                        retrieval_level="topic",
                        metadata={
                            "entity_name": entity.entity_name,
                            "entity_type": entity.entity_type,
                            "matched_keyword": keyword,
                            "source": "category_match"
                        }
                    ))
        
        # 3. 如果结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_topic_level_search(topic_keywords, top_k - len(results))
            results.extend(neo4j_results)
            
        # 4. 按相关性排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"主题级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]
    
    def _neo4j_topic_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j主题级检索补充"""
        results = []
        
        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (r:Recipe)
                WHERE r.category CONTAINS keyword 
                   OR r.cuisineType CONTAINS keyword
                   OR r.tags CONTAINS keyword
                WITH r, keyword
                OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
                WITH r, keyword, collect(i.name)[0..3] as ingredients
                RETURN 
                    r.nodeId as node_id,
                    r.name as name,
                    r.category as category,
                    r.cuisineType as cuisine_type,
                    r.difficulty as difficulty,
                    ingredients,
                    keyword as matched_keyword
                ORDER BY r.difficulty ASC, r.name
                LIMIT $limit
                """
                
                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })
                
                for record in result:
                    content_parts = []
                    content_parts.append(f"菜品: {record['name']}")
                    
                    if record["category"]:
                        content_parts.append(f"分类: {record['category']}")
                    if record["cuisine_type"]:
                        content_parts.append(f"菜系: {record['cuisine_type']}")
                    if record["difficulty"]:
                        content_parts.append(f"难度: {record['difficulty']}")
                    
                    if record["ingredients"]:
                        ingredients_str = ', '.join(record["ingredients"][:3])
                        content_parts.append(f"主要食材: {ingredients_str}")
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=record["node_id"],
                        node_type="Recipe",
                        relevance_score=0.75,  # 补充检索得分
                        retrieval_level="topic",
                        metadata={
                            "name": record["name"],
                            "category": record["category"],
                            "cuisine_type": record["cuisine_type"],
                            "difficulty": record["difficulty"],
                            "matched_keyword": record["matched_keyword"],
                            "source": "neo4j_fallback"
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Neo4j主题级检索失败: {e}")
            
        return results
        
    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """
        双层检索：结合实体级和主题级检索
        """
        logger.info(f"开始双层检索: {query}")
        
        # 1. 提取关键词
        entity_keywords, topic_keywords = self.extract_query_keywords(query)
        
        # 2. 执行双层检索
        entity_results = self.entity_level_retrieval(entity_keywords, top_k)
        topic_results = self.topic_level_retrieval(topic_keywords, top_k)
        
        # 3. 结果合并和排序
        all_results = entity_results + topic_results
        
        # 4. 去重和重排序
        seen_nodes = set()
        unique_results = []
        
        for result in sorted(all_results, key=lambda x: x.relevance_score, reverse=True):
            if result.node_id not in seen_nodes:
                seen_nodes.add(result.node_id)
                unique_results.append(result)
        
        # 5. 转换为Document格式
        documents = []
        for result in unique_results[:top_k]:
            # 确保recipe_name字段正确设置
            recipe_name = result.metadata.get("name") or result.metadata.get("entity_name", "未知菜品")
            
            doc = Document(
                page_content=result.content,
                metadata={
                    "node_id": result.node_id,
                    "node_type": result.node_type,
                    "retrieval_level": result.retrieval_level,
                    "relevance_score": result.relevance_score,
                    "recipe_name": recipe_name,  # 确保有recipe_name字段
                    "search_type": "dual_level",  # 设置搜索类型
                    **result.metadata
                }
            )
            documents.append(doc)
            
        logger.info(f"双层检索完成，返回 {len(documents)} 个文档")
        return documents
    
    def vector_search_enhanced(self, query: str, top_k: int = 5) -> List[Document]:
        """
        增强的向量检索：结合图信息
        """
        try:
            # 使用Milvus进行向量检索
            vector_docs = self.milvus_module.similarity_search(query, k=top_k*2)
            
            # 用图信息增强结果并转换为Document对象
            enhanced_docs = []
            for result in vector_docs:
                # 从Milvus结果创建Document对象
                content = result.get("text", "")
                metadata = result.get("metadata", {})
                node_id = metadata.get("node_id")
                
                if node_id:
                    # 从图中获取邻居信息
                    neighbors = self._get_node_neighbors(node_id)
                    if neighbors:
                        # 将邻居信息添加到内容中
                        neighbor_info = f"\n相关信息: {', '.join(neighbors[:3])}"
                        content += neighbor_info
                
                # 确保recipe_name字段正确设置
                recipe_name = metadata.get("recipe_name", "未知菜品")
                
                # 调试：打印向量得分
                vector_score = result.get("score", 0.0)
                logger.debug(f"向量检索得分: {recipe_name} = {vector_score}")
                
                # 创建Document对象
                doc = Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "recipe_name": recipe_name,  # 确保有recipe_name字段
                        "score": vector_score,
                        "search_type": "vector_enhanced"
                    }
                )
                enhanced_docs.append(doc)
                
            return enhanced_docs[:top_k]
            
        except Exception as e:
            logger.error(f"增强向量检索失败: {e}")
            return []
    
    def _get_node_neighbors(self, node_id: str, max_neighbors: int = 3) -> List[str]:
        """获取节点的邻居信息"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n {nodeId: $node_id})-[r]-(neighbor)
                RETURN neighbor.name as name
                LIMIT $limit
                """
                result = session.run(query, {"node_id": node_id, "limit": max_neighbors})
                return [record["name"] for record in result if record["name"]]
        except Exception as e:
            logger.error(f"获取邻居节点失败: {e}")
            return []
    
    def bm25_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        BM25 检索：jieba 分词后查 BM25Okapi 索引，按分数降序返回 top_k。
        分数写入 metadata["bm25_score"]，供调试与未来潜在的分数级融合使用。
        """
        if self.bm25 is None or not self.bm25_corpus_docs:
            logger.warning("BM25 索引未初始化，bm25_search 返回空")
            return []

        tokenized_query = self._tokenize_chinese(query)
        if not tokenized_query:
            logger.debug(f"BM25 query 分词为空，跳过: {query}")
            return []

        scores = self.bm25.get_scores(tokenized_query)
        # 按分数降序取 top_k 索引
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        docs: List[Document] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                # BM25 分数 ≤ 0 视为无关（IDF/TF 全无贡献），不进结果
                continue
            src = self.bm25_corpus_docs[idx]
            recipe_name = (
                src.metadata.get("recipe_name")
                or src.metadata.get("name")
                or "未知菜品"
            )
            doc = Document(
                page_content=src.page_content,
                metadata={
                    **src.metadata,
                    "recipe_name": recipe_name,
                    "search_method": "bm25",
                    "search_type": "bm25",
                    "bm25_score": score,
                }
            )
            docs.append(doc)

        logger.info(f"BM25 检索完成，返回 {len(docs)} 个文档（query tokens={tokenized_query}）")
        return docs

    @staticmethod
    def _rrf_merge(
        ranked_lists: List[Tuple[str, List[Document]]],
        top_k: int,
        k: int = _RRF_K,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion: score(d) = Σ_i 1 / (k + best_rank_i(d))

        Args:
            ranked_lists: 多路 (source_name, ranked_docs) — docs 按相关度降序
            top_k: 最终返回个数
            k: RRF 平滑常数，默认 60（Cormack et al. 2009）

        去重 key：node_id 优先，page_content[:200] hash 兜底。

        同 source 内同 doc_id 多次命中（如一道菜的多个 chunk 共享 recipe.nodeId）：
            - 算分只取该 source 内最佳 rank（最小 rank）一次，避免重复加分
            - 命中 chunk 数另存到 rrf_chunk_hits，供后续分析

        canonical doc（最终展示给 LLM 的 page_content）：
            选全局最小 rank 那个 chunk；rank 相同时按 ranked_lists 顺序优先。

        返回的 Document 是新对象，不会 mutate 输入 list 里的 Document。
        """
        # doc_id -> source_name -> 该 source 内最小 rank（用于算分）
        best_rank_per_source: Dict[str, Dict[str, int]] = {}
        # doc_id -> source_name -> 该 source 内命中 chunk 次数（信息存档）
        chunk_hits_per_source: Dict[str, Dict[str, int]] = {}
        # doc_id -> (global_best_rank, source_priority, doc) — 选 canonical doc
        best_doc_info: Dict[str, Tuple[int, int, Document]] = {}

        for source_priority, (source_name, ranked_docs) in enumerate(ranked_lists):
            for rank, doc in enumerate(ranked_docs, start=1):
                node_id = doc.metadata.get("node_id")
                doc_id = (
                    str(node_id) if node_id is not None
                    else f"hash::{hashlib.md5(doc.page_content[:200].encode('utf-8')).hexdigest()}"
                )

                if doc_id not in best_rank_per_source:
                    best_rank_per_source[doc_id] = {}
                    chunk_hits_per_source[doc_id] = {}

                curr_best = best_rank_per_source[doc_id].get(source_name)
                # 如果是第一次出现或者当前rank比记录的更小，则更新
                if curr_best is None or rank < curr_best:
                    best_rank_per_source[doc_id][source_name] = rank

                chunk_hits_per_source[doc_id][source_name] = (
                    chunk_hits_per_source[doc_id].get(source_name, 0) + 1
                )

                new_key = (rank, source_priority)
                if (
                    doc_id not in best_doc_info
                    or new_key < (best_doc_info[doc_id][0], best_doc_info[doc_id][1])
                ):
                    best_doc_info[doc_id] = (rank, source_priority, doc)

        # 每个 source 只用 best rank 算一次贡献
        rrf_scores: Dict[str, float] = {
            doc_id: sum(1.0 / (k + r) for r in source_ranks.values())
            for doc_id, source_ranks in best_rank_per_source.items()
        }

        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True
        )

        merged: List[Document] = []
        for doc_id in sorted_ids[:top_k]:
            _, _, source_doc = best_doc_info[doc_id]
            # 浅 copy metadata，避免 mutate 上游 Document
            new_metadata = dict(source_doc.metadata)
            new_metadata["rrf_score"] = rrf_scores[doc_id]
            new_metadata["rrf_sources"] = list(best_rank_per_source[doc_id].keys())
            new_metadata["rrf_ranks"] = dict(best_rank_per_source[doc_id])
            new_metadata["rrf_chunk_hits"] = dict(chunk_hits_per_source[doc_id])
            new_metadata["final_score"] = rrf_scores[doc_id]
            merged.append(Document(
                page_content=source_doc.page_content,
                metadata=new_metadata,
            ))

        return merged

    def _build_parent_doc_map(self) -> Dict[str, Document]:
        """{str(node_id): 整篇父菜谱 Document}，由分块前的 data_module.documents 懒建一次。"""
        docs = getattr(self.data_module, "documents", None) or []
        m: Dict[str, Document] = {}
        for d in docs:
            nid = d.metadata.get("node_id")
            if nid is not None:
                m[str(nid)] = d
        return m

    def _attach_parent_documents(self, docs: List[Document]) -> List[Document]:
        """RRF 去重后，前 parent_doc_top_n 条且能在映射中找到父菜谱的，
        用整篇父菜谱（超 parent_doc_max_chars 截断）替换 chunk；其余原样不变。
        不改顺序/数量/排名，不 mutate 输入（被替换的造新 Document，未替换的直接传原对象）。"""
        if getattr(self, "_parent_doc_map", None) is None:
            self._parent_doc_map = self._build_parent_doc_map()
        pmap = self._parent_doc_map
        if not pmap:
            logger.warning("父文档映射为空（data_module.documents 未就绪），父文档回填未生效，仍然使用原chunk填充上下文")
            return docs
        top_n = getattr(self.config, "parent_doc_top_n", 3)
        max_chars = getattr(self.config, "parent_doc_max_chars", 4000)

        out: List[Document] = []
        for i, doc in enumerate(docs):
            if i >= top_n:
                out.append(doc)
                continue
            nid = doc.metadata.get("node_id")
            key = str(nid if nid is not None else doc.metadata.get("parent_id"))
            parent = pmap.get(key)
            if parent is None:
                out.append(doc)
                continue
            pc = parent.page_content or ""
            if len(pc) > max_chars:
                pc = pc[:max_chars] + "…（父文档已截断）"
            out.append(Document(page_content=pc, metadata=dict(doc.metadata)))
        return out

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        混合检索：三路召回（图键值双层 + 向量 + BM25）→ RRF 融合
        """
        logger.info(f"开始混合检索（dual + vector + bm25, RRF k={_RRF_K}）: {query}")

        # 每路给 RRF 留够候选空间，否则三路各自前 top_k 容易没交集，融合退化
        candidate_k = max(top_k * 2, 10)

        dual_docs = self.dual_level_retrieval(query, candidate_k)
        vector_docs = self.vector_search_enhanced(query, candidate_k)
        bm25_docs = self.bm25_search(query, candidate_k)

        # 标记每路来源（dual_level 内部会写 search_type 但不一定写 search_method）
        for d in dual_docs:
            d.metadata.setdefault("search_method", "dual_level")
        for d in vector_docs:
            d.metadata["search_method"] = "vector"
        # bm25_search 内部已写 search_method=bm25

        final_docs = self._rrf_merge(
            ranked_lists=[
                ("dual_level", dual_docs),
                ("vector", vector_docs),
                ("bm25", bm25_docs),
            ],
            top_k=top_k,
        )

        # 父文档回填（仅 hybrid_traditional 路；不改排名，仅换上下文内容）
        if getattr(self.config, "enable_parent_doc_retrieval", False):
            final_docs = self._attach_parent_documents(final_docs)

        logger.info(
            f"RRF 融合完成：dual={len(dual_docs)} vector={len(vector_docs)} "
            f"bm25={len(bm25_docs)} → 最终 {len(final_docs)} 个文档"
        )
        return final_docs

    def close(self):
        """关闭资源连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭") 