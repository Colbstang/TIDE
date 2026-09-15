"""Public checks: no article text needed, and none may enter the ZIP."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]
from reproduce import PRIVATE_TEXT, package, verify_inputs, verify_outputs
from source import CASES, analysis_mask, load_vectors


class PublicTests(unittest.TestCase):
    def test_vectors_align_with_public_pmid_manifest_and_primary_tau(self):
        with (ROOT / "out/records.csv").open() as handle:
            manifest = list(csv.DictReader(handle))
        for case in CASES:
            vectors, years, anchor = load_vectors(case)
            rows = [r for r in manifest if r["case"] == case["case"]]
            self.assertEqual(len(rows), len(years))
            self.assertEqual(len({r["pmid"] for r in rows}), len(rows))
            np.testing.assert_array_equal(years, [int(r["year"]) for r in rows])
            np.testing.assert_array_equal(analysis_mask(years), [bool(int(r["analyzed"])) for r in rows])
            self.assertEqual([int(r["vector_row"]) for r in rows], list(range(len(rows))))
            scores = np.sum(vectors.astype(float) * anchor.astype(float), axis=1)
            bins = np.unique(years[analysis_mask(years)])
            means = [scores[years == y].mean() for y in bins]
            signs = [np.sign(means[j] - means[i]) for i in range(len(bins)) for j in range(i + 1, len(bins))]
            self.assertAlmostEqual(round(sum(signs) / len(signs), 3), case["observed_tau"])

    def test_package_omits_text_even_when_present_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "public.zip"
            with patch("builtins.print"):
                package(destination)
            with zipfile.ZipFile(destination) as z:
                names = {n.removeprefix("TIDE/") for n in z.namelist()}
                self.assertFalse(names & PRIVATE_TEXT)
                self.assertFalse(any(n.startswith("data/raw/") for n in names))
                self.assertIn("out/records.csv", names)

    def test_private_text_is_optional_only_in_public_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = next(iter(PRIVATE_TEXT))
            (root / "provenance.json").write_text(json.dumps({"artifacts": {name: "0" * 64}}))
            with patch("reproduce.ROOT", root), patch("builtins.print"):
                verify_inputs(with_text=False)
                with self.assertRaisesRegex(SystemExit, "separate frozen text archive"):
                    verify_inputs(with_text=True)

    def test_public_output_verification_rejects_missing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SystemExit, "missing generated output"):
                verify_outputs(Path(directory), False, public_only=True)


if __name__ == "__main__":
    unittest.main()
