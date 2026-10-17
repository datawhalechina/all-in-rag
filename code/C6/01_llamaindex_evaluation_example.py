from dotenv import load_dotenv
# 从 .env 文件加载环境变量（例如 DEEPSEEK_API_KEY）
load_dotenv()

import os
import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner,
)
from llama_index.core.evaluation.eval_utils import get_results_df
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset

# 以脚本所在目录为基准构造路径，避免依赖运行时的当前工作目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(SCRIPT_DIR, "../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf")
DATASET_PATH = os.path.join(SCRIPT_DIR, "c6_response_eval_dataset.json")

# 全局配置：LlamaIndex 的 Settings 是全局单例，
# 之后所有 Index / QueryEngine 默认使用这里的 LLM 和嵌入模型。
# temperature=0.1 让回答更稳定，便于评估对比。
Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
# 注意：文档是英文 PDF，所以用英文嵌入模型 bge-small-en；
# 若换成中文语料，应改用 bge-small-zh 之类。
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")


async def main():
    # 1. 加载文档
    # 直接读取指定 PDF，SimpleDirectoryReader 会按页解析成 Document 列表。
    reader = SimpleDirectoryReader(input_files=[PDF_PATH])
    documents = reader.load_data()

    # 1.1 加载或生成响应评估数据集
    # 评估需要「问题 + 参考答案」对。若本地已有缓存 JSON 就直接加载，
    # 避免每次运行都重新调用 LLM 生成数据集（耗时且花钱）。
    if os.path.exists(DATASET_PATH):
        print("加载响应评估数据集...")
        response_eval_dataset = QueryResponseDataset.from_json(DATASET_PATH)
    else:
        print("生成响应评估数据集...")
        # 只取前 10 个 Document 生成问题，控制成本；
        # DatasetGenerator 会调用 LLM 根据文档内容自动出题并生成参考答案。
        dataset_generator = DatasetGenerator.from_documents(documents[:10])  # 减少文档数量
        # num=15 表示生成 15 个问答对；减少数量以控制生成时间和 token 消耗。
        response_eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=15)  # 减少问题数量
        # 保存到本地，下次运行直接复用
        response_eval_dataset.save_json(DATASET_PATH)

    # 2. 构建两种不同的RAG查询引擎和检索器进行对比
    # 目的：对比「句子窗口检索」与「常规分块检索」在回答质量上的差异。

    # 2.1 句子窗口检索
    # SentenceWindowNodeParser 的做法：
    #   - 索引粒度是「单句」（每个 Node 只含一句），保证检索精准；
    #   - 但每个 Node 的 metadata 里额外存一个 window（前后各 window_size 句）；
    #   - 检索命中某句后，用 MetadataReplacementPostProcessor 把
    #     Node 的文本替换成 window 内容，再喂给 LLM，保证上下文完整。
    # 这就是「小粒度检索 + 大粒度生成」的典型思路。
    sentence_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,                              # 前后各取 3 句组成窗口
        window_metadata_key="window",               # 窗口文本存在 metadata 的这个 key
        original_text_metadata_key="original_text", # 原始单句存在这个 key
    )
    sentence_nodes = sentence_parser.get_nodes_from_documents(documents)
    sentence_index = VectorStoreIndex(sentence_nodes)

    # 查询引擎：检索 top-2 个句子节点，然后用窗口文本替换后再生成回答。
    sentence_query_engine = sentence_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )
    # 检索器：只做检索、不生成，这里实际未在评估中使用（评估用的是 query_engine）。
    sentence_retriever = sentence_index.as_retriever(similarity_top_k=2)

    # 2.2 常规分块检索（基准）
    # 传统做法：按固定 chunk_size=512 切块，检索和生成都用同一粒度。
    # 作为对照基线，用来衡量句子窗口检索是否真的更好。
    base_parser = SentenceSplitter(chunk_size=512)
    base_nodes = base_parser.get_nodes_from_documents(documents)
    base_index = VectorStoreIndex(base_nodes)

    base_query_engine = base_index.as_query_engine(similarity_top_k=2)
    base_retriever = base_index.as_retriever(similarity_top_k=2)

    # 3. 初始化响应评估器
    # FaithfulnessEvaluator：回答是否忠实于检索到的上下文（有无编造）。
    # RelevancyEvaluator：回答是否与问题相关。
    # 两者都调用 LLM 做判断，会产生额外 API 调用。
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)

    # 4. 执行响应评估对比
    print("开始执行响应评估对比...")
    evaluators = {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator}
    queries = response_eval_dataset.queries

    # 句子窗口检索响应评估
    # BatchEvalRunner 并发执行多个 query 的评估；workers=2 控制并发度，
    # 避免同时打太多请求触发 API 限流。
    print("\n=== 评估句子窗口检索 ===")
    sentence_runner = BatchEvalRunner(evaluators, workers=2, show_progress=True)
    sentence_response_results = await sentence_runner.aevaluate_queries(
        queries=queries, query_engine=sentence_query_engine
    )

    # 常规分块检索响应评估
    print("\n=== 评估常规分块检索 ===")
    base_runner = BatchEvalRunner(evaluators, workers=2, show_progress=True)
    base_response_results = await base_runner.aevaluate_queries(
        queries=queries, query_engine=base_query_engine
    )

    # 5. 分析并打印对比结果
    print("\n" + "="*60)
    print("响应评估结果对比")
    print("="*60)

    def calc_response_score(results, metric):
        """计算某个评估指标的通过率。

        results 是 BatchEvalRunner 返回的 dict，形如
        {"faithfulness": [EvaluationResult, ...], "relevancy": [...]}。
        每个 EvaluationResult 有 passing 字段（True/False），
        这里用「通过数 / 总数」得到通过率。
        """
        if results and results.get(metric):
            scores = results[metric]
            return sum(r.passing for r in scores) / len(scores)
        return 0

    # 句子窗口检索结果
    sentence_faith = calc_response_score(sentence_response_results, "faithfulness")
    sentence_rel = calc_response_score(sentence_response_results, "relevancy")

    # 常规分块检索结果
    base_faith = calc_response_score(base_response_results, "faithfulness")
    base_rel = calc_response_score(base_response_results, "relevancy")

    print(f"\n句子窗口检索:")
    print(f"  忠实度: {sentence_faith:.1%}")
    print(f"  相关性: {sentence_rel:.1%}")

    print(f"\n常规分块检索:")
    print(f"  忠实度: {base_faith:.1%}")
    print(f"  相关性: {base_rel:.1%}")

    # 简单对比
    # 注意：15 个问题样本量很小，且 LLM 评估本身有随机性，
    # 结论只能作为趋势参考，不能当作严格实验结论。
    if sentence_faith > base_faith and sentence_rel > base_rel:
        print(f"\n✅ 句子窗口检索在两个维度上都优于常规分块检索")
    elif sentence_faith > base_faith or sentence_rel > base_rel:
        print(f"\n⚖️  句子窗口检索在某些维度上有优势")
    else:
        print(f"\n❌ 句子窗口检索未显示明显优势")


if __name__ == "__main__":
    # 整个流程是 async 的（数据集生成、批量评估都是异步接口），
    # 所以用 asyncio.run 启动事件循环。
    asyncio.run(main())