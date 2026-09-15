"""The publication index must be calculated, uniquely keyed, and fail closed."""
import csv
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from paper import append, core


class PaperTests(unittest.TestCase):
    def test_calculated_counts_and_query_summary(self):
        with patch("paper.OUT", str(ROOT / "out")), patch("paper.save_csv") as save:
            core()
        rows = save.call_args.args[1]
        values = {(r["case"], r["metric"]): float(r["value"]) for r in rows}
        self.assertEqual(len(values), len(rows))
        for key, expected in [(('three_cases', 'case_record_observations'), 4504),
                              (('three_cases', 'distinct_pmids'), 4503),
                              (('DRF', 'included_years'), 24), (('PJI', 'last_year_n'), 364),
                              (('ACS', 'query_unique_definitions'), 4),
                              (('PJI', 'query_nominal_significant'), 4),
                              (('DRF', 'query_nominal_significant'), 3)]:
            self.assertEqual(values[key], expected)
        self.assertLess(values['DRF', 'matched_tfidf_paraphrase_tau'], 0)
        self.assertGreater(values['DRF', 'mean_paraphrase_tau'], 0)

    def test_exact_monte_carlo_probabilities(self):
        for filename, probability, count, reps in [
            ('case_study_metrics.csv', 'yearshuffle_exact_p', 'yearshuffle_exceedances', 'yearshuffle_reps'),
            ('robustness_null.csv', 'phase_exact_p', 'exceedances', 'replicates')]:
            with (ROOT / 'out' / filename).open() as handle:
                for row in csv.DictReader(handle):
                    self.assertEqual(float(row[probability]), (int(row[count])+1)/(int(row[reps])+1))

    def test_nonfinite_index_rejected(self):
        for value in (float('nan'), float('inf'), '-inf'):
            with self.assertRaises(ValueError):
                append([], 'PJI', 'test', value, 'test')

    def test_annual_centroid_identity(self):
        with (ROOT / 'out/annual_series.csv').open() as handle:
            for row in csv.DictReader(handle):
                product = float(row['centroid_norm']) * float(row['centroid_anchor_cosine'])
                self.assertAlmostEqual(float(row['mean_cosine']), product, places=6)
