import unittest

from src.run import Retriever, rrf_scores, top_indices


class RetrievalTests(unittest.TestCase):
    def test_stable_ranking(self):
        self.assertEqual(top_indices([1.0, 3.0, 3.0], 3), [1, 2, 0])

    def test_rrf_rewards_agreement(self):
        scores = rrf_scores(
            rankings=[[0, 1, 2], [1, 0, 2]],
            size=3,
            constant=60,
        )
        self.assertAlmostEqual(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])

    def test_bm25_retrieves_matching_document(self):
        corpus = [
            {"doc_id": "a", "title": "Fruit", "text": "apple orchard fruit"},
            {"doc_id": "b", "title": "Sea", "text": "ocean water waves"},
            {"doc_id": "c", "title": "Space", "text": "planet orbit stars"},
        ]
        retriever = Retriever(
            corpus=corpus,
            config={},
            method="bm25",
            corpus_hash="test",
            cache_dir=".cache/test",
        )

        ranked = retriever.retrieve("apple", k=10)
        self.assertEqual(ranked[0], "a")
        self.assertEqual(len(ranked), 3)
        self.assertEqual(len(ranked), len(set(ranked)))

    def test_closed_book_has_no_retrieval(self):
        retriever = Retriever(
            corpus=[],
            config={},
            method="closed_book",
            corpus_hash="test",
            cache_dir=".cache/test",
        )
        self.assertEqual(retriever.retrieve("Any question?", 10), [])


if __name__ == "__main__":
    unittest.main()