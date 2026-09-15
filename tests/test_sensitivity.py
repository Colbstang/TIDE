"""Independent checks of evidence weighting and query-input integrity."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from sensitivity import METHODS, evidence_weight, trend, yearly_series
from source import CASES, load_vectors, primary_records
from query_sensitivity import audit, normalized_response, packed, selected_data


class WeightTests(unittest.TestCase):
    def test_exact_publication_type_rule(self):
        self.assertEqual(evidence_weight([]), 1.0)
        self.assertEqual(evidence_weight(["Journal Article", "Clinical Trial", "Randomized Controlled Trial"]), 3.0)
        for name in ["Pragmatic Clinical Trial", "Clinical Trial, Phase III", "Clinical Trial, Phase IV"]:
            self.assertEqual(evidence_weight([name]), 3.0)
        for name in ["Observational Study", "Evaluation Study", "Validation Study"]:
            self.assertEqual(evidence_weight([name]), 1.5)
        self.assertEqual(evidence_weight(["Clinical Trial, Phase II"]), 1.0)
        # Original max(..., 1.0) made its declared 0.5 Case Reports weight inert.
        self.assertEqual(evidence_weight(["Case Reports"]), 1.0)

    def test_weighted_mean_independent_and_calendar_gate(self):
        years = np.repeat([2000, 2001, 2002, 2003, 2004, 2025], 6)
        sims = np.arange(len(years), dtype=float) / len(years)
        weights = np.tile([1, 1, 1.5, 2, 2.5, 3], 6)
        rows = yearly_series(years, sims, weights)
        self.assertEqual(len(rows), 5)
        for row in rows:
            idx = [i for i, y in enumerate(years) if y == row["year"]]
            expected = sum(float(weights[i]) * float(sims[i]) for i in idx) / sum(weights[i] for i in idx)
            self.assertAlmostEqual(row["weighted_mean"], expected, places=15)

    def test_invalid_input_fails(self):
        years = np.repeat(np.arange(2000, 2005), 5)
        sims = np.arange(25) / 25
        for values, weights in [(sims[:-1], None), (sims, np.zeros(25)),
                                (np.full(25, np.nan), None), (sims, np.full(25, np.inf))]:
            with self.assertRaises(ValueError):
                yearly_series(years, values, weights)
        with self.assertRaises(ValueError):
            yearly_series(years + .5, sims)
        with self.assertRaises(ValueError):
            yearly_series(years[:20], sims[:20])

    def test_weighted_tau_independent_pair_count(self):
        expected_current = {"PJI": -0.7794117647058824, "ACS": -.06, "DRF": .4275362318840579}
        for case in CASES:
            av, years, gv = load_vectors(case)
            _, selected = primary_records(case)
            weights = np.array([evidence_weight(r.get("publication_types", [])) for _, r in selected])
            cosines = np.einsum("ij,j->i", av.astype(float), gv.astype(float))
            rows = yearly_series(years, cosines, weights)
            values = [r["weighted_mean"] for r in rows]
            tau = sum(np.sign(values[j]-values[i]) for i in range(len(values))
                      for j in range(i+1,len(values))) / (len(values)*(len(values)-1)/2)
            self.assertAlmostEqual(tau, trend(rows)[0], places=12)
            self.assertAlmostEqual(tau, expected_current[case["case"]], places=12)



class QueryTests(unittest.TestCase):
    def test_record_and_text_counts_are_distinct(self):
        _, records, _ = selected_data(ROOT / "data/sensitivity")
        from source import text_of
        spec = json.loads((ROOT / "provenance.json").read_text())["sensitivity"]["query_reanalysis"]
        self.assertEqual(len(records), spec["selected_unique_records"])
        self.assertEqual(len({text_of(r) for r in records}), spec["distinct_selected_texts"])

    def test_fresh_query_tau_independent_pair_count(self):
        directory = ROOT / "data/sensitivity"
        selections, _, _ = selected_data(directory)
        with (directory / "query_cosines.csv").open() as handle:
            scores = {r["pmid"]: r for r in csv.DictReader(handle)}
        with (ROOT / "out/query_sensitivity.csv").open() as handle:
            reference = {(r["case"], r["variant"]): r for r in csv.DictReader(handle)}
        for query, records in selections:
            bins = {}
            for record in records:
                if 2000 <= int(record["year"]) <= 2024:
                    bins.setdefault(int(record["year"]), []).append(float(scores[record["pmid"]][query["case"]]))
            retained = [bins[y] for y in sorted(bins) if len(bins[y]) >= 5]
            means = [sum(values)/len(values) for values in retained]
            signs = [np.sign(means[j]-means[i]) for i in range(len(means)) for j in range(i+1,len(means))]
            tau = sum(signs) / np.sqrt(len(signs)*sum(s != 0 for s in signs))
            saved = reference[query["case"], query["variant"]]
            self.assertAlmostEqual(tau, float(saved["tau"]), places=12)
            self.assertEqual(sum(map(len, retained)), int(saved["analyzed_records"]))
            self.assertEqual(len(retained), int(saved["n_years"]))

    def test_books_accounted_for_without_becoming_article_inputs(self):
        xml = b'<PubmedArticleSet><PubmedBookArticle><BookDocument><PMID>42</PMID></BookDocument></PubmedBookArticle></PubmedArticleSet>'
        rows = normalized_response(xml)
        self.assertEqual(rows[0]["pmid"], "42")
        self.assertEqual(rows[0]["record_type"], "PubmedBookArticle")
        from source import select_records
        self.assertEqual(select_records(rows), [])
        with self.assertRaises(ValueError):
            normalized_response(b'<PubmedArticleSet><ERROR>Failed</ERROR></PubmedArticleSet>')

    def test_exact_three_case_five_variant_scope(self):
        self.assertEqual({q["case"] for q in METHODS["queries"]}, {"PJI", "ACS", "DRF"})
        for case in ["PJI", "ACS", "DRF"]:
            variants = [q["variant"] for q in METHODS["queries"] if q["case"] == case]
            self.assertEqual(len(variants), 5)
            self.assertEqual(len(set(variants)), 5)

    def test_query_memberships_and_text_scores_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            records = [dict(pmid=str(i+1), year=2000+i//5, title=f"Record {i}", abstract="",
                            languages=["eng"], publication_types=["Journal Article"]) for i in range(25)]
            manifest = dict(methods_sha256=hashlib.sha256((ROOT / "data/sensitivity/methods.json").read_bytes()).hexdigest(),
                            completed_at_utc="test", queries=[dict(case=q["case"], variant=q["variant"],
                            pmids=[r["pmid"] for r in records]) for q in METHODS["queries"]])
            packed(directory / "query_records.json.gz", records)
            packed(directory / "query_memberships.json.gz", manifest)
            self.assertEqual(len(selected_data(directory)[1]), 25)
            score_text = "pmid,text_sha256,PJI,ACS,DRF\n" + "".join(
                f"{r['pmid']},{hashlib.sha256(r['title'].encode()).hexdigest()},{i/30},{i/30},{i/30}\n"
                for i,r in enumerate(records))
            (directory / "query_cosines.csv").write_text(score_text)
            with patch("query_sensitivity.save_csv") as save, patch("sys.stdout", new_callable=io.StringIO):
                audit(directory)
                self.assertEqual(len(save.call_args_list[0].args[1]), 15)
            (directory / "query_cosines.csv").write_text(score_text.replace(records[0]["pmid"]+",", "999,", 1))
            with self.assertRaises(ValueError):
                audit(directory)
            manifest["queries"][0]["pmids"].append("1")
            packed(directory / "query_memberships.json.gz", manifest)
            with self.assertRaises(ValueError):
                selected_data(directory)
