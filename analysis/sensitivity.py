"""Evidence weighting and annual aggregation for the three-case manuscript."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kendalltau, theilslopes

from common import ROOT
from source import CASES, analysis_mask, load_vectors, primary_records, save_csv

METHODS = json.loads((Path(ROOT) / "data/sensitivity/methods.json").read_text())


def evidence_weight(publication_types):
    """Maximum exact publication-type weight, with a 1.0 floor."""
    return max([1.0] + [METHODS["evidence_weights"].get(pt, 1.0)
                        for pt in publication_types])


def yearly_series(years, similarities, weights=None):
    """Annual means on >=5 records/year within 2000–2024; reject invalid input."""
    years, similarities = np.asarray(years), np.asarray(similarities)
    weights = np.ones(len(years)) if weights is None else np.asarray(weights)
    if not (years.ndim == similarities.ndim == weights.ndim == 1 and
            len(years) == len(similarities) == len(weights)):
        raise ValueError("Years, similarities and weights must be aligned vectors")
    if not (np.isfinite(years).all() and np.isfinite(similarities).all() and
            np.isfinite(weights).all() and (weights > 0).all()):
        raise ValueError("Nonfinite data or nonpositive evidence weights")
    if not np.equal(years, years.astype(int)).all():
        raise ValueError("Noninteger publication year")
    rows = []
    for y in np.unique(years):
        keep = years == y
        if keep.sum() < 5 or not 2000 <= y <= 2024:
            continue
        values, w = similarities[keep], weights[keep]
        rows.append(dict(year=int(y), n=int(keep.sum()), weight_sum=float(w.sum()),
                         mean=float(np.mean(values)),
                         weighted_mean=float(np.average(values, weights=w))))
    if len(rows) < 5:
        raise ValueError("Fewer than five eligible annual bins; trend not estimable")
    return rows


def trend(rows, column="weighted_mean"):
    years = [r["year"] for r in rows]
    means = [r[column] for r in rows]
    result = kendalltau(years, means)
    if not np.isfinite([result.statistic, result.pvalue]).all():
        raise ValueError("Degenerate annual trend")
    return float(result.statistic), float(result.pvalue), float(theilslopes(means, years).slope)


def weighting():
    """Apply the declared weights to the primary anchors and analysis years."""
    results, annual, records = [], [], []
    for case in CASES:
        _, selected = primary_records(case)
        av, years, gv = load_vectors(case)
        np.testing.assert_array_equal(years, [int(r["year"]) for _, r in selected])
        weights = np.array([evidence_weight(r.get("publication_types", [])) for _, r in selected])
        cosines = np.einsum("ij,j->i", av.astype(float), gv.astype(float))
        current = yearly_series(years, cosines, weights)
        tau, p, slope = trend(current)
        results.append(dict(case=case["case"], n=sum(r["n"] for r in current), n_years=len(current),
                            primary_tau=trend(current, "mean")[0], weighted_tau=tau,
                            weighted_p=p, weighted_slope_per_year=slope))
        annual.extend(dict(case=case["case"], **r) for r in current)
        keep = analysis_mask(years)
        records.extend(dict(case=case["case"], vector_row=i, pmid=str(r["pmid"]),
                            year=int(years[i]), analyzed=int(keep[i]), weight=float(weights[i]))
                       for i, (_, r) in enumerate(selected))
        print(f"{case['case']}: weighted tau={tau:+.4f}", flush=True)
    save_csv("evidence_weighting.csv", results)
    save_csv("evidence_weighting_annual.csv", annual)
    save_csv("evidence_weights.csv", records)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "weighting":
        raise SystemExit("Usage: python analysis/sensitivity.py weighting")
    weighting()
