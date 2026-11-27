"""Offline regression tests for chunk identity during reciprocal-rank fusion."""
import sys
import unittest
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
spec = importlib.util.spec_from_file_location(
    "retrieval_optimization", Path(__file__).resolve().parents[1] / "rag_modules/retrieval_optimization.py",
)
retrieval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieval_module)
RetrievalOptimizationModule = retrieval_module.RetrievalOptimizationModule


class FixtureEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class TestChunkIdentity(unittest.TestCase):
    def setUp(self):
        self.docs = [Document(page_content="盐 适量", metadata={
            "chunk_id": source + "_0", "source": source,
        }) for source in ("A", "B")]
        self.retrieval = RetrievalOptimizationModule(
            FAISS.from_documents(self.docs, FixtureEmbeddings()), self.docs,
        )

    def test_hybrid_retains_distinct_chunks_with_identical_content(self):
        results = self.retrieval.hybrid_search("盐", top_k=5)
        self.assertEqual({doc.metadata["chunk_id"] for doc in results}, {"A_0", "B_0"})

    def test_metadata_filter_can_find_either_source(self):
        for source in ("A", "B"):
            with self.subTest(source=source):
                results = self.retrieval.metadata_filtered_search("盐", {"source": source})
                self.assertEqual([doc.metadata["source"] for doc in results], [source])

    def test_same_chunk_fuses_scores_across_retrievers(self):
        a = self.docs[0]
        copy = Document(page_content=a.page_content, metadata=dict(a.metadata))
        results = self.retrieval._rrf_rerank([a], [copy])
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].metadata["rrf_score"], 2 / 61)

    def test_legacy_chunks_without_ids_still_fuse_by_content(self):
        a = Document(page_content="legacy")
        b = Document(page_content="legacy")
        results = self.retrieval._rrf_rerank([a], [b])
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].metadata["rrf_score"], 2 / 61)

    def test_explicit_id_cannot_collide_with_a_legacy_content_digest(self):
        import hashlib
        legacy = Document(page_content="legacy")
        identified = Document(page_content="different", metadata={
            "chunk_id": hashlib.md5(b"legacy").hexdigest(),
        })
        self.assertEqual(len(self.retrieval._rrf_rerank([legacy], [identified])), 2)

    def test_empty_retriever_results(self):
        self.assertEqual(self.retrieval._rrf_rerank([], []), [])


if __name__ == "__main__":
    unittest.main()
