import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rag.evidence import prepare_evidence, source_label
from rag.evaluate import DEFAULT_DATASET, evaluate, load_cases, score_sources


def doc(text, source, page=None):
    metadata = {"source": source}
    if page is not None:
        metadata["page"] = page
    return SimpleNamespace(page_content=text, metadata=metadata)


class EvidenceTests(unittest.TestCase):
    def test_empty_and_duplicate_sources(self):
        context, sources = prepare_evidence([
            doc(" ", "empty.txt"),
            doc("第一段", r"C:\data\维护保养.txt"),
            doc("第二段", r"C:\data\维护保养.txt"),
        ])
        self.assertEqual(sources, ["维护保养.txt"])
        self.assertIn("第一段", context)
        self.assertIn("第二段", context)
        self.assertNotIn("empty.txt", context)

    def test_pdf_page_is_one_based(self):
        self.assertEqual(source_label({"source": "/data/manual.pdf", "page": 0}), "manual.pdf（第1页）")
        self.assertEqual(prepare_evidence([]), ("", []))


class RetrievalEvaluationTests(unittest.TestCase):
    def test_hit_and_rank(self):
        self.assertEqual(score_sources(["a.txt", "b.txt"], ["b.txt"]),
                         {"hit": True, "rank": 2, "reciprocal_rank": 0.5})
        self.assertEqual(score_sources(["a.txt"], ["b.txt"]),
                         {"hit": False, "rank": None, "reciprocal_rank": 0.0})

    def test_evaluate_with_fake_retriever(self):
        cases = [
            {"id": "one", "query": "a", "expected_sources": ["expected.txt"]},
            {"id": "two", "query": "b", "expected_sources": ["missing.txt"]},
        ]
        report = evaluate(cases, lambda _query, _k: [
            doc("irrelevant", "other.txt"),
            doc("relevant", "expected.txt"),
        ], 2)
        self.assertEqual(report["hit_at_k"], 0.5)
        self.assertEqual(report["mrr_at_k"], 0.25)
        self.assertEqual(report["cases"][0]["rank"], 2)
        with self.assertRaises(ValueError):
            evaluate(cases, lambda _query, _k: [], 0)

    def test_sample_cases_reference_existing_files(self):
        for case in load_cases(DEFAULT_DATASET):
            for source in case["expected_sources"]:
                self.assertTrue((DEFAULT_DATASET.parent.parent / "data" / source).is_file(), case["id"])

    def test_duplicate_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text('{"id":"x","query":"q","expected_sources":["a.txt"]}\n'
                            '{"id":"x","query":"q","expected_sources":["a.txt"]}\n',
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path)


if __name__ == "__main__":
    unittest.main()
