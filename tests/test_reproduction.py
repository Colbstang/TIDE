"""Small, model-free regression and independently calculated statistics checks."""
import itertools
import io
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree as ET
import zipfile

import numpy as np
from scipy.stats import kendalltau, theilslopes, t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
from source import (CASES, EXCLUDE, analysis_mask, load_vectors, primary_analysis_records,
                    primary_records, select_records, text_of)
from drift import annual_mean_cos, boot_same_sign, fig_series
from robustness import cluster_robust_ols, ols_year_adj, phase_surrogate, series
from pubmed import normalize_article, normalize_response, search_ids
from reproduce import (equivalent, package, SOURCE_FILES, CORE_NUMERIC, MODEL_NUMERIC,
                       CORE_FIGURES, MODEL_FIGURES, main as reproduce_main)
import tempfile
from collections import Counter
import re
from boundary import PAIRS
import boundary


def record(**changes):
    return {"pmid": "1", "title": "  Title  ", "abstract": " Text ", "year": 2000,
            "languages": ["eng"], "publication_types": ["Journal Article"], **changes}


class SourceTests(unittest.TestCase):
    def test_reproduction_rejects_reference_overwrite(self):
        with patch.object(sys, "argv", ["reproduce.py", "--write"]), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                reproduce_main()
        self.assertEqual(result.exception.code, 2)

    def test_text_and_title_only(self):
        self.assertEqual(text_of(record()), "Title Text")
        self.assertEqual(text_of(record(abstract=None)), "Title")

    def test_english_aliases_and_missing_language(self):
        for lang in ["eng", "English", " en "]:
            self.assertEqual(len(select_records([record(languages=lang)])), 1)
        for lang in [None, [], ["spa"]]:
            self.assertEqual(select_records([record(languages=lang)]), [])

    def test_all_exclusions(self):
        for excluded in EXCLUDE:
            self.assertEqual(select_records([record(publication_types=["Journal Article", excluded])]), [])

    def test_order_and_required_fields(self):
        records = [record(pmid="7"), record(pmid="8", year=None), record(pmid="9", year=2025),
                   record(pmid=""), record(pmid="10", title="", abstract="")]
        self.assertEqual([i for i, _ in select_records(records)], [0, 2])

    def test_publication_year_cannot_be_silently_truncated(self):
        for year in (2000, "2000", 2000.0):
            self.assertEqual(len(select_records([record(year=year)])), 1)
        for year in (2000.9, 1999.999, True, False):
            with self.assertRaisesRegex(ValueError, "Noninteger publication year"):
                select_records([record(year=year)])

    def test_duplicate_is_failure(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_records([record(), record()])

    def test_window_and_floor(self):
        years = np.array([1999] * 5 + [2000] * 5 + [2001] * 4 + [2024] * 6 + [2025] * 5)
        self.assertEqual(int(analysis_mask(years).sum()), 11)

    def test_every_archived_row(self):
        self.assertEqual([c["case"] for c in CASES], ["PJI", "ACS", "DRF"])
        for case in CASES:
            _, selected = primary_records(case)
            av, years, gv = load_vectors(case)
            np.testing.assert_array_equal(years, [int(r["year"]) for _, r in selected])
            self.assertEqual(len(set(str(r["pmid"]) for _, r in selected)), len(av))
            self.assertEqual(int(analysis_mask(years).sum()), case["n_abstracts"])
            np.testing.assert_allclose(np.linalg.norm(av, axis=1), 1, atol=2e-7)
            # Two explicit float64 reductions avoid platform-specific float32 BLAS warnings.
            scalar = np.sum(av.astype(float) * gv.astype(float), axis=1)
            np.testing.assert_allclose(scalar, np.einsum("ij,j->i", av.astype(float), gv.astype(float)), atol=5e-7)

    def test_xml_normalization(self):
        xml = '<PubmedArticle><PMID>42</PMID><Article><ArticleTitle>A <i>title</i></ArticleTitle><Journal><Title>J</Title><JournalIssue><PubDate><MedlineDate>2001 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal><Abstract><AbstractText>First <b>part</b></AbstractText><AbstractText>Second</AbstractText></Abstract><Language>eng</Language><PublicationTypeList><PublicationType>Clinical Trial</PublicationType></PublicationTypeList></Article></PubmedArticle>'
        r = normalize_article(ET.fromstring(xml))
        self.assertEqual((r["pmid"], r["year"], r["title"], r["abstract"]), ("42", 2001, "A title", "First part Second"))
        self.assertEqual(r["publication_types"], ["Clinical Trial"])

    def test_response_accounts_for_book_records(self):
        xml = b'<PubmedArticleSet><PubmedBookArticle><BookDocument><PMID>42</PMID></BookDocument></PubmedBookArticle></PubmedArticleSet>'
        rows = normalize_response(xml)
        self.assertEqual(rows, [{"pmid": "42", "record_type": "PubmedBookArticle",
                                "exclusion": "Not a PubmedArticle; excluded by original ingestion rule"}])
        with self.assertRaises(ValueError):
            normalize_response(b'<PubmedArticleSet><ERROR>Failed</ERROR></PubmedArticleSet>')

    def test_search_rejects_truncation_duplicates_errors(self):
        self.assertEqual(search_ids({"esearchresult": {"count": "1", "idlist": ["42"]}}), ["42"])
        for result in [{"count": "2", "idlist": ["42"]}, {"count": "2", "idlist": ["42", "42"]},
                       {"count": "0", "idlist": []}, {"count": "1", "idlist": ["42"], "errorlist": {"bad": "query"}}]:
            with self.assertRaises(ValueError):
                search_ids({"esearchresult": result})


class StatisticTests(unittest.TestCase):
    def test_context_probe_stops_after_document_reaching_cap(self):
        def tokenizer(text, **kwargs):
            return {"input_ids": np.ones((1, int(text)), dtype=int)}

        def model(**kwargs):
            hidden = np.ones((len(kwargs["input_ids"][0]), 2))
            return SimpleNamespace(last_hidden_state=[SimpleNamespace(numpy=lambda: hidden)])

        with patch.dict(sys.modules, {"torch": SimpleNamespace(no_grad=nullcontext)}), \
                patch.dict(boundary._M, {"atok": tokenizer, "amod": model}):
            self.assertEqual(len(boundary.term_in_context(["1"] * 200, [1])), 200)
            self.assertEqual(len(boundary.term_in_context(["1"] * 199 + ["2", "2"], [1])), 201)

    def test_primary_cosines_use_float64_reduction(self):
        for case in CASES:
            years, cosines = annual_mean_cos({"d": case["trial_id"]})
            self.assertEqual(cosines.dtype, np.dtype("float64"))
            self.assertTrue(np.isfinite(cosines).all())
            self.assertEqual(len(years), len(cosines))

    def test_volume_adjustment_independent(self):
        # Residualize year against log-volume, independently of the matrix solver.
        for case in CASES:
            years, means, counts = series(case["trial_id"], 5)
            volume = np.array([counts[int(y)] for y in years], float)
            x = (years - years.mean()) / years.std()
            z = np.log(volume)
            z = (z - z.mean()) / z.std()
            rx = x - np.sum(x*z) / np.sum(z*z) * z
            beta = np.sum(rx * means) / np.sum(rx*rx)
            residual = means - means.mean() - beta*x
            residual -= np.sum(residual*z) / np.sum(z*z) * z
            se = np.sqrt(np.sum(residual**2) / (len(years)-3) / np.sum(rx*rx))
            actual_beta, actual_p, _, _ = ols_year_adj(years, means, volume)
            self.assertAlmostEqual(actual_beta, beta, places=12)
            self.assertAlmostEqual(actual_p, 2*student_t.sf(abs(beta/se), len(years)-3), places=10)

    def test_minimal_pairs_preserve_word_multisets(self):
        self.assertEqual(set(PAIRS), {"PJI", "DRF"})
        for pairs in PAIRS.values():
            for a, b in pairs:
                self.assertEqual(Counter(re.findall(r"[a-z]+", a.lower())), Counter(re.findall(r"[a-z]+", b.lower())))

    def test_primary_tau_and_slope_independent(self):
        for case in CASES:
            av, years, gv = load_vectors(case)
            sims = np.einsum("ij,j->i", av.astype(float), gv.astype(float))
            ys = np.unique(years[analysis_mask(years)])
            means = [sims[years == y].mean() for y in ys]
            pairs = list(itertools.combinations(range(len(ys)), 2))
            tau = sum(np.sign(means[j] - means[i]) for i, j in pairs) / len(pairs)
            slopes = [(means[j] - means[i]) / (ys[j] - ys[i]) for i, j in pairs]
            self.assertAlmostEqual(tau, kendalltau(ys, means).statistic, places=12)
            self.assertAlmostEqual(round(tau, 3), case["observed_tau"], places=3)
            self.assertAlmostEqual(np.median(slopes), theilslopes(means, ys).slope, places=12)

    def test_bootstrap_conditions_on_year(self):
        # Separated, internally constant yearly values cannot change sign.
        ys = np.repeat([2000, 2001, 2002], 5)
        np.random.seed(0)
        self.assertEqual(boot_same_sign([2000, 2001, 2002], ys, ys.astype(float), 1, B=20), 100)

    def test_pji_percentile_interval_independently(self):
        values = fig_series(CASES[0]["trial_id"])
        years = sorted(values)
        point = kendalltau(years, [values[y].mean() for y in years]).statistic
        rng = np.random.RandomState(0)
        draws = [kendalltau(years, [rng.choice(values[y], len(values[y]), replace=True).mean()
                                   for y in years]).statistic for _ in range(1500)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        np.testing.assert_allclose([lo, hi], [-0.823529411764706, -0.5294117647058824], atol=1e-12)
        self.assertLess(point, lo)  # Preserve the actual percentile result, not a forced-containing interval.

    def test_phase_preserves_power_spectrum(self):
        for n in [17, 24, 25]:
            x = np.random.default_rng(3).normal(size=n)
            surrogate = phase_surrogate(x, np.random.default_rng(4))
            np.testing.assert_allclose(np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(surrogate)), atol=1e-12)

    def test_cluster_sandwich_independent(self):
        # Centering year is algebraically equivalent and improves conditioning.
        for case in CASES:
            av, years, gv = load_vectors(case)
            keep = analysis_mask(years)
            y = years[keep]
            response = np.einsum("ij,j->i", av.astype(float), gv.astype(float))[keep]
            x = y.astype(float) - y.mean()
            slope = np.sum(x * (response - response.mean())) / np.sum(x*x)
            actual, p = cluster_robust_ols(y, response)
            self.assertAlmostEqual(slope, actual, places=9)
            residual = response - response.mean() - slope * x
            groups = np.unique(y)
            meat = sum(np.sum(x[y == g] * residual[y == g])**2 for g in groups)
            correction = len(groups) / (len(groups)-1) * (len(y)-1) / (len(y)-2)
            se = np.sqrt(correction * meat / np.sum(x*x)**2)
            expected_p = 2 * student_t.sf(abs(slope / se), len(groups)-1)
            self.assertAlmostEqual(p, expected_p, places=7)

    def test_comparator_fails_changed_schema_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.csv", Path(directory) / "b.csv"
            a.write_text("case,value\nPJI,0.42\n")
            for content in ["case,value\nACS,0.42\n", "case,value\nPJI,0.43\n",
                            "case,value\nPJI,nan\n", "wrong,value\nPJI,0.42\n"]:
                b.write_text(content)
                self.assertFalse(equivalent(a, b, atol=1e-5))
            b.write_text("case,value\nPJI,0.420001\n")
            self.assertFalse(equivalent(a, b))
            self.assertTrue(equivalent(a, b, atol=1e-5))
            # A difference exactly on the declared boundary must not fail
            # because its binary representation lands a few ulps above it.
            b.write_text("case,value\nPJI,0.42001000000000001\n")
            self.assertTrue(equivalent(a, b, atol=1e-5))
            b.write_text("case,value\nPJI,0.4200101\n")
            self.assertFalse(equivalent(a, b, atol=1e-5))
            a.write_text("case,weighted_slope_per_year\nPJI,0.42\n")
            b.write_text("case,weighted_slope_per_year\nPJI,0.420001\n")
            self.assertTrue(equivalent(a, b, atol=1e-5))
            a.write_text("pmid,year,value\n123,2024,0.42\n")
            for content in ["pmid,year,value\n123.0,2024,0.42\n",
                            "pmid,year,value\n123,2024.000001,0.42\n"]:
                b.write_text(content)
                self.assertFalse(equivalent(a, b, atol=1e-5))

    def test_comparator_rejects_invalid_data_even_when_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.csv", Path(directory) / "b.csv"
            for content in ["", "case,value\nPJI,nan\n", "case,value\nPJI,inf\n",
                            "case,value\nPJI,-inf\n", "case,value\nPJI\n",
                            "case,value\nPJI,0.42,5\n", "case,value,value\nPJI,.1,.2\n",
                            "case,\nPJI,.1\n"]:
                a.write_text(content)
                b.write_text(content)
                for tolerance in (0, 1e-5):
                    self.assertFalse(equivalent(a, b, atol=tolerance), content)
            for left, right in [
                ("case,value\nPJI,0.42,5\n", "case,value\nPJI,0.42,999\n"),
                ("case,value,extra\nPJI,0.42\n", "case,value,extra\nPJI,0.420001\n"),
                ("case,value,other\nPJI,nan,0.42\n", "case,value,other\nPJI,nan,0.420001\n"),
            ]:
                a.write_text(left)
                b.write_text(right)
                self.assertFalse(equivalent(a, b, atol=1e-5))
            a, b = Path(directory) / "a.npy", Path(directory) / "b.npy"
            for value in (np.nan, np.inf, -np.inf):
                np.save(a, [value])
                np.save(b, [value])
                self.assertFalse(equivalent(a, b))
                self.assertFalse(equivalent(a, b, atol=1e-5))

    def test_package_excludes_unlisted_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required = (*SOURCE_FILES, "README.md", "LICENSE", "CITATION.cff", "anchors.json",
                        "reproduce.py", "requirements.txt", "requirements-core.txt", ".gitignore",
                        ".gitattributes", ".github/workflows/reproduce.yml")
            for name in (*required, "analysis/private_note.py", "tests/scratch.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test fixture\n")
            (root / "provenance.json").write_text(json.dumps(dict(artifacts={}, reference_artifacts={})))
            destination = root / "candidate.zip"
            with patch("reproduce.ROOT", root), patch("builtins.print"):
                package(destination)
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(set(archive.namelist()),
                                 {"TIDE/" + name for name in (*required, "provenance.json")})

    def test_every_reference_has_a_reproduction_path(self):
        provenance = json.loads((ROOT / "provenance.json").read_text())
        outputs = CORE_NUMERIC + MODEL_NUMERIC + CORE_FIGURES + MODEL_FIGURES
        self.assertEqual(len(outputs), len(set(outputs)))
        self.assertEqual(set(provenance["reference_artifacts"]),
                         {"out/" + name for name in outputs})
        sources = {str(p.relative_to(ROOT)) for folder in ("analysis", "tests")
                   for p in (ROOT / folder).glob("*.py") if not p.name.startswith(".")}
        self.assertEqual(set(SOURCE_FILES), sources)

    def test_interpretability_uses_exact_primary_rows(self):
        for case in CASES:
            rows = primary_analysis_records(case["case"])
            self.assertEqual(len(rows), case["n_abstracts"])
            self.assertTrue(all(2000 <= year <= 2024 and text for year, text in rows))


if __name__ == "__main__":
    unittest.main()
