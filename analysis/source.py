"""Trace archived records to primary vectors; rebuild and verify with pinned MedCPT.

No original application or live PubMed call is needed. `audit` is model-free;
`rebuild` additionally re-encodes every record using the pinned model revisions.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter

import numpy as np
from scipy.stats import kendalltau

from common import EMB, RAW, ROOT, OUT, load_medcpt

CONFIG = json.loads((Path(ROOT) / "anchors.json").read_text())
CASES = CONFIG["cases"]
FILTERS = CONFIG["corpus_filters"]
EXCLUDE = {v.lower() for v in FILTERS["exclude_publication_types"]}
VECTOR_ATOL = 1e-5


def text_of(record):
    """The historical primary encoder's exact title/abstract concatenation."""
    return " ".join(p for p in ((record.get("title") or "").strip(),
                              (record.get("abstract") or "").strip()) if p)


def select_records(records):
    """Return (raw-row, record) pairs, preserving order; reject ambiguous PMIDs."""
    selected, seen = [], set()
    for row, record in enumerate(records):
        langs = record.get("languages") or []
        if isinstance(langs, str):
            langs = [langs]
        if not {str(v).strip().lower() for v in langs} & {"en", "eng", "english"}:
            continue
        types = {v.strip().lower() for v in (record.get("publication_types") or [])
                 if isinstance(v, str)}
        if types & EXCLUDE:
            continue
        pmid = str(record.get("pmid") or "").strip()
        year = record.get("year")
        try:
            integer_year = int(year)
        except (ValueError, TypeError):
            continue
        # int() truncates fractional numbers and treats booleans as years.
        if isinstance(year, bool) or (not isinstance(year, str) and year != integer_year):
            raise ValueError(f"Noninteger publication year: {year!r}")
        if not pmid or not text_of(record):
            continue
        if pmid in seen:
            raise ValueError(f"Duplicate retained PMID: {pmid}")
        seen.add(pmid)
        selected.append((row, record))
    return selected


def primary_records(case):
    path = Path(RAW) / f"{case['trial_id']}_pubmed_2000_2024.json"
    raw = json.loads(path.read_text())
    selected = select_records(raw)
    if len(raw) != case["n_raw"] or len(selected) != case["n_vectors"]:
        raise ValueError(f"{case['case']}: source count differs from archived specification")
    return raw, selected


def primary_analysis_records(case_name):
    """Return exact primary-analysis (year, text) rows for one configured case."""
    case = next(spec for spec in CASES if spec["case"] == case_name)
    _, selected = primary_records(case)
    records = [record for _, record in selected]
    years = np.array([int(record["year"]) for record in records])
    keep = analysis_mask(years)
    return [(int(year), text_of(record)) for year, record, retained
            in zip(years, records, keep) if retained]


def analysis_mask(years):
    counts = Counter(years)
    return np.array([2000 <= y <= 2024 and counts[y] >= FILTERS["min_abstracts_per_year"]
                     for y in years], dtype=bool)


def unit(v):
    v = np.asarray(v)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.isfinite(v).all() or not np.isfinite(norm).all() or (norm == 0).any():
        raise ValueError("Embedding contains nonfinite values or zero norm")
    return v / norm


def load_vectors(case):
    folder = Path(EMB) / case["trial_id"]
    av = np.load(folder / "abstract_vectors.npy", allow_pickle=False)
    years = np.load(folder / "years.npy", allow_pickle=False)
    gv = np.load(folder / "guideline_vector.npy", allow_pickle=False).ravel()
    if av.shape != (case["n_vectors"], 768) or years.shape != (len(av),) or gv.shape != (768,):
        raise ValueError(f"{case['case']}: invalid embedding shapes")
    return unit(av), years, unit(gv)


def save_csv(name, rows):
    with (Path(OUT) / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit():
    """Model-free row alignment, completeness, yearly means, and matched baseline."""
    from drift import tfidf_pubcount_evid
    manifest, counts, annual, baselines = [], [], [], []
    for case in CASES:
        raw, selected = primary_records(case)
        records = [r for _, r in selected]
        av, years, gv = load_vectors(case)
        expected_years = np.array([int(r["year"]) for r in records])
        np.testing.assert_array_equal(years, expected_years)
        keep = analysis_mask(years)
        ys = np.unique(years[keep])
        if int(keep.sum()) != case["n_abstracts"] or len(ys) != case["n_years"]:
            raise ValueError(f"{case['case']}: analyzed counts differ")
        sims = np.einsum("ij,j->i", av.astype(float), gv.astype(float))
        for row, (raw_row, r) in enumerate(selected):
            manifest.append(dict(case=case["case"], vector_row=row, raw_row=raw_row,
                                 pmid=str(r["pmid"]), year=int(years[row]), analyzed=int(keep[row]),
                                 text_sha256=hashlib.sha256(text_of(r).encode()).hexdigest()))
        counts.append(dict(case=case["case"], raw_records=len(raw), excluded_before_embedding=len(raw)-len(records),
                           embedded_records=len(records),
                           title_only_records=sum(not (r.get("abstract") or "").strip() for r in records),
                           title_only_analyzed=sum(not (r.get("abstract") or "").strip() for r, k in zip(records, keep) if k),
                           outside_window=int(((years < 2000) | (years > 2024)).sum()),
                           below_year_floor=int(((years >= 2000) & (years <= 2024) & ~keep).sum()),
                           analyzed_records=int(keep.sum()), analyzed_years=len(ys)))
        for y in ys:
            v = sims[years == y]
            centroid = av[years == y].astype(float).mean(0)
            norm = np.linalg.norm(centroid)
            # Mean cosine combines centroid direction and within-year concentration.
            angle = v.mean() / (norm * np.linalg.norm(gv.astype(float)))
            annual.append(dict(case=case["case"], year=int(y), n=len(v),
                               mean_cosine=round(float(v.mean()), 8),
                               standard_error=round(float(v.std(ddof=1) / np.sqrt(len(v))), 8),
                               centroid_norm=round(float(norm), 8), centroid_anchor_cosine=round(float(angle), 8)))
        selected_in_analysis = [r for r, k in zip(records, keep) if k]
        tf, vol, ev, pct = tfidf_pubcount_evid(
            {"d": case["trial_id"], "text": case["anchor_text"]}, selected_in_analysis)
        baselines.append(dict(case=case["case"], n=int(keep.sum()), n_years=len(ys),
                              tfidf_tau=round(tf, 6), pubcount_tau=round(vol, 6),
                              evidence_volume_tau=round(ev, 6), percent_high_evidence=round(pct, 3)))
    for name, rows in [("records.csv", manifest), ("source_counts.csv", counts),
                       ("annual_series.csv", annual), ("matched_baselines.csv", baselines)]:
        save_csv(name, rows)
    print(f"PASS: {len(manifest)} unique case/PMID rows aligned; primary counts and years match", flush=True)


def encode_mean(tokenizer, model, texts, batch_size=16, return_cls=False):
    """Attention-mask mean including special tokens; max 512; L2 normalization.

    This reproduces the historical SentenceTransformer fallback, not MedCPT's
    native CLS pooling. Batch padding tokens contribute zero to the mean.
    """
    import torch
    chunks, cls_chunks = [], []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            means = (hidden * mask).sum(1) / mask.sum(1)
            chunks.append(torch.nn.functional.normalize(means, dim=1).numpy())
            if return_cls:
                cls_chunks.append(torch.nn.functional.normalize(hidden[:, 0, :], dim=1).numpy())
    result = np.concatenate(chunks)
    return (result, np.concatenate(cls_chunks)) if return_cls else result


def rebuild():
    """Re-encode all rows and anchors, write new vectors only inside TIDE_OUT."""
    audit()
    qtok, qmod, atok, amod = load_medcpt()
    from sensitivity import METHODS
    legacy = np.load(Path(ROOT) / "data/sensitivity/legacy_drf_anchor.npy", allow_pickle=False).ravel()
    rebuilt_legacy = encode_mean(qtok, qmod, [METHODS["legacy_drf_anchor"]])[0]
    np.testing.assert_allclose(rebuilt_legacy, unit(legacy), atol=VECTOR_ATOL, rtol=0)
    save_csv("legacy_anchor_reconstruction.csv", [dict(case="DRF", max_abs_error=float(
        np.max(np.abs(rebuilt_legacy - unit(legacy)))), tolerance=VECTOR_ATOL)])
    report, cls_report = [], []
    for case in CASES:
        _, selected = primary_records(case)
        old, years, old_anchor = load_vectors(case)
        print(f"Rebuilding {case['case']}: {len(selected)} records", flush=True)
        new, cls_vectors = encode_mean(atok, amod, [text_of(r) for _, r in selected], return_cls=True)
        anchors, cls_anchors = encode_mean(qtok, qmod, [case["anchor_text"]], return_cls=True)
        anchor = anchors[0]
        # Normalize both sides: the archived DRF anchor has a nonunit scale.
        np.testing.assert_allclose(new, old, atol=VECTOR_ATOL, rtol=0)
        np.testing.assert_allclose(anchor, old_anchor, atol=VECTOR_ATOL, rtol=0)
        old_cos = np.einsum("ij,j->i", old.astype(float), old_anchor.astype(float))
        new_cos = np.einsum("ij,j->i", unit(new).astype(float), unit(anchor).astype(float))
        np.testing.assert_allclose(new_cos, old_cos, atol=VECTOR_ATOL, rtol=0)
        ys = np.unique(years[analysis_mask(years)])
        old_mean = [old_cos[years == y].mean() for y in ys]
        new_mean = [new_cos[years == y].mean() for y in ys]
        old_tau, new_tau = kendalltau(ys, old_mean).statistic, kendalltau(ys, new_mean).statistic
        np.testing.assert_allclose(new_tau, old_tau, atol=1e-12, rtol=0)
        cls_cos = np.einsum("ij,j->i", cls_vectors.astype(float), cls_anchors[0].astype(float))
        cls_tau = kendalltau(ys, [cls_cos[years == y].mean() for y in ys]).statistic
        cls_report.append(dict(case=case["case"], n=int(analysis_mask(years).sum()), n_years=len(ys),
                               mean_tau=round(float(new_tau), 6), matched_cls_tau=round(float(cls_tau), 6)))
        folder = Path(OUT) / "rebuilt" / case["trial_id"]
        folder.mkdir(parents=True, exist_ok=True)
        for name, value in [("abstract_vectors", new), ("guideline_vector", anchor), ("years", years)]:
            np.save(folder / f"{name}.npy", value, allow_pickle=False)
        report.append(dict(case=case["case"], rows=len(new),
                           max_vector_abs_error=float(np.max(np.abs(new - old))),
                           max_anchor_abs_error=float(np.max(np.abs(anchor - old_anchor))),
                           max_cosine_abs_error=float(np.max(np.abs(new_cos - old_cos))),
                           original_tau=float(old_tau), rebuilt_tau=float(new_tau), tolerance=VECTOR_ATOL))
        print(f"PASS {case['case']}: all rows and anchor match; tau={new_tau:+.12f}", flush=True)
    save_csv("reconstruction.csv", report)
    save_csv("matched_cls.csv", cls_report)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "audit"
    {"audit": audit, "rebuild": rebuild}[command]()
