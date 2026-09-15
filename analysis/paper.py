"""Generate the manuscript's numeric index; never transcribe reported results.

`core` needs only frozen vectors. `probes` runs after model reconstruction.
Detailed annual/token values remain in their source tables, not duplicated here.
"""
import csv
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kendalltau

from common import OUT, ROOT
from source import CASES, analysis_mask, primary_records, load_vectors, save_csv, unit
from sensitivity import METHODS
from drift import tfidf_pubcount_evid


def read(name):
    with (Path(OUT) / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def append(rows, case, metric, value, source):
    if not np.isfinite(float(value)):
        raise ValueError(f"Nonfinite manuscript value: {case}/{metric}")
    rows.append(dict(case=case, metric=metric, value=value, source=source))


def index_tables(rows, names):
    """Expose numeric cells with stable case/variant keys and source-column names."""
    for name in names:
        identifiers = ("variant", "min_count_floor", "direction", "fit_unit", "score_unit")
        # Only the cross-anchor table uses anchor as a case ID, not prose.
        if name == "cross_anchors.csv":
            identifiers += ("anchor",)
        for row in read(name):
            key = row["case"]
            for field in identifiers:
                if field in row:
                    key += "/" + row[field]
            for field, value in row.items():
                if field in ("case", "variant", "direction", "expected", "years", "method"):
                    continue
                try:
                    float(value)
                except ValueError:
                    continue
                append(rows, key, f"{Path(name).stem}.{field}", value, f"{name}:{field}")


def core():
    rows = []
    records = [r for r in read("records.csv") if int(r["analyzed"])]
    append(rows, "three_cases", "case_record_observations", len(records), "records.csv:analyzed=1")
    append(rows, "three_cases", "distinct_pmids", len({r["pmid"] for r in records}), "records.csv:pmid,analyzed=1")
    for case in CASES:
        name = case["case"]
        annual = [r for r in read("annual_series.csv") if r["case"] == name]
        counts = [int(r["n"]) for r in annual]
        summary = dict(included_years=len(annual), first_year=annual[0]["year"],
                       last_year=annual[-1]["year"], first_year_n=counts[0], last_year_n=counts[-1],
                       annual_n_min=min(counts), annual_n_median=float(np.median(counts)), annual_n_max=max(counts))
        for field, value in summary.items():
            append(rows, name, field, value, "annual_series.csv:year,n")
        for field in ("centroid_norm", "centroid_anchor_cosine"):
            tau = kendalltau([int(r["year"]) for r in annual], [float(r[field]) for r in annual]).statistic
            append(rows, name, field + "_tau", float(tau), "annual_series.csv:" + field)
        query = [r for r in read("query_sensitivity.csv") if r["case"] == name]
        taus = [float(r["tau"]) for r in query]
        summary = dict(query_labels=len(query), query_unique_definitions=len({
            q["query"] for q in METHODS["queries"] if q["case"] == name}),
            query_tau_min=min(taus), query_tau_max=max(taus),
            query_negative=sum(t < 0 for t in taus), query_positive=sum(t > 0 for t in taus),
            query_nominal_significant=sum(float(r["nominal_p"]) < .05 for r in query))
        for field, value in summary.items():
            append(rows, name, field, value, "query_sensitivity.csv; methods.json:queries")
    # Check the manuscript's lexical-anchor claim using both explicitly defined inputs.
    case = next(c for c in CASES if c["case"] == "DRF")
    _, selected = primary_records(case)
    vectors, years, _ = load_vectors(case)
    old_anchor = unit(np.load(Path(ROOT) / "data/sensitivity/legacy_drf_anchor.npy").ravel())
    scores = np.einsum("ij,j->i", vectors.astype(float), old_anchor.astype(float))
    included_years = np.unique(years[analysis_mask(years)])
    tau = kendalltau(included_years, [scores[years == y].mean() for y in included_years]).statistic
    append(rows, "DRF", "mean_paraphrase_tau", float(tau), "legacy_drf_anchor.npy; primary article vectors/years")
    selected = [r for (_, r), keep in zip(selected, analysis_mask(years)) if keep]
    tau = tfidf_pubcount_evid(dict(d=case["trial_id"], text=METHODS["legacy_drf_anchor"]), selected)[0]
    append(rows, "DRF", "matched_tfidf_paraphrase_tau", tau,
           "drift.tfidf_pubcount_evid; methods.json:legacy_drf_anchor")
    index_tables(rows, ["source_counts.csv", "case_study_metrics.csv", "drift_resampling.csv",
                       "robustness_ols.csv", "robustness_loo.csv", "robustness_null.csv",
                       "robustness_voladj.csv", "robustness_bincount.csv", "matched_baselines.csv",
                       "evidence_weighting.csv", "query_sensitivity.csv",
                       "tfidf_variants.csv", "cross_anchors.csv"])
    save_csv("paper_core.csv", rows)


def probes():
    rows = []
    index_tables(rows, ["matched_cls.csv", "geometry_summary.csv", "minimal_pairs.csv"])
    pairs = read("minimal_pairs.csv")
    for name in dict.fromkeys(r["case"] for r in pairs):
        pair = {r["direction"]: r for r in pairs if r["case"] == name}
        for method in ("medcpt", "tfidf"):
            field = method + "_cosine"
            delta = float(pair["concordant"][field]) - float(pair["discordant"][field])
            append(rows, name, method + "_minimal_pair_delta", round(delta, 6), f"minimal_pairs.csv:{field}")
    # Figure 3 uses a centered three-bin mean with reflected endpoint padding.
    from interpretability import WORD_SHOW, _smooth
    annual = read("word_traj_PJI.csv")
    for term, _, _ in WORD_SHOW["PJI"]["terms"]:
        values = np.array([float(r[term]) for r in annual])
        for label, series in [("raw", values), ("smoothed", _smooth(values))]:
            append(rows, "PJI", f"{term}_{label}_peak_year", annual[int(np.argmax(series))]["year"],
                   f"word_traj_PJI.csv:{term}")
    save_csv("paper_probes.csv", rows)


if __name__ == "__main__":
    {"core": core, "probes": probes}[sys.argv[1]]()
