"""Independent checks for the matched lexical and cross-anchor comparisons."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from comparisons import components, comparisons, lexical_variants
from unittest.mock import patch
from source import CASES
from drift import fig_series
from scipy.stats import kendalltau


class ComparisonTests(unittest.TestCase):
    def test_numeric_index_preserves_probe_keys_and_distinguishes_anchor_cases(self):
        from paper import index_tables
        tables = {
            "minimal_pairs.csv": [dict(case="DRF", direction="concordant",
                                       anchor="Clinical anchor prose.", medcpt_cosine="0.8")],
            "cross_anchors.csv": [dict(case="DRF", anchor=name, tau="0.4")
                                  for name in ("PJI", "ACS", "DRF")],
        }
        rows = []
        with patch("paper.read", side_effect=tables.__getitem__):
            index_tables(rows, list(tables))
        self.assertEqual([row["case"] for row in rows],
                         ["DRF/concordant", "DRF/PJI", "DRF/ACS", "DRF/DRF"])

    def test_comparisons_match_independent_audit_and_primary_diagonal(self):
        with patch("comparisons.save_csv") as save:
            comparisons()
        tables = {call.args[0]: call.args[1] for call in save.call_args_list}
        lexical = tables["tfidf_variants.csv"]
        cross = tables["cross_anchors.csv"]
        self.assertEqual(len(lexical), 12)
        self.assertEqual(len(cross), 9)
        expected = {"PJI": -.44117647, "ACS": -.02, "DRF": .63043478}
        for row in lexical:
            if (row["fit_unit"], row["score_unit"]) == ("record", "record_mean"):
                self.assertAlmostEqual(row["tau"], expected[row["case"]], places=8)
        for case in CASES:
            bins = fig_series(case["trial_id"])
            direct = kendalltau(sorted(bins), [bins[y].mean() for y in sorted(bins)]).statistic
            row = next(r for r in cross if r["case"] == r["anchor"] == case["case"])
            self.assertAlmostEqual(row["tau"], direct, places=8)
        self.assertTrue(all(r["tau"] > 0 for r in cross if r["case"] == "DRF"))
        # Both representations must retain every eligible annual observation.
        self.assertEqual(len(tables["tfidf_annual.csv"]), 4 * sum(c["n_years"] for c in CASES))
        self.assertEqual(len(tables["cross_anchor_annual.csv"]), 3 * sum(c["n_years"] for c in CASES))

    def test_lexical_scoring_averages_records_not_concatenated_documents(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = ["alpha alpha beta", "beta gamma", "alpha gamma", "gamma gamma beta"]
        years = np.array([2000, 2000, 2001, 2001])
        variants = {(fit, score): values for fit, score, _, values in lexical_variants(texts, years, "alpha beta")}
        vectorizer = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)
        matrix = vectorizer.fit_transform(["alpha beta"] + texts).toarray()
        direct = matrix[1:] @ matrix[0]
        np.testing.assert_allclose(variants["record", "record_mean"], direct.reshape(2, 2).mean(1))
        self.assertFalse(np.allclose(variants["record", "record_mean"], variants["record", "year"]))
        with self.assertRaises(ValueError):
            list(lexical_variants(texts, years[:-1], "alpha beta"))

    def test_centroid_identity_and_concentration(self):
        vectors = np.array([[1., 0.], [0., 1.], [1., 0.], [1., 0.]])
        years = np.array([2000, 2000, 2001, 2001])
        ys, _, means, length, angle = components(vectors, years, np.array([1., 0.]))
        np.testing.assert_array_equal(ys, [2000, 2001])
        np.testing.assert_allclose(means, [.5, 1.], atol=1e-12, rtol=0)
        np.testing.assert_allclose(length, [np.sqrt(.5), 1.], atol=1e-12, rtol=0)
        np.testing.assert_allclose(angle, [np.sqrt(.5), 1.], atol=1e-12, rtol=0)
