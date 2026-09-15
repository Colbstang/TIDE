"""Three fixed anchors, five recovered retrieval definitions per case.

fetch NEW_DIRECTORY archives a fresh PubMed retrieval; it never replaces history.
score DIRECTORY encodes each selected PMID once and saves three cosines.
audit [DIRECTORY] recomputes query trends from saved records, memberships and scores.
rebuild [DIRECTORY] verifies all saved scores directly from their archived text.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
import numpy as np

from common import ROOT, load_medcpt
from pubmed import normalize_response as normalized_response, request, search_ids
from sensitivity import METHODS, trend, yearly_series
from source import CASES, EXCLUDE, encode_mean, save_csv, select_records, text_of

DATA = Path(ROOT) / "data/sensitivity"


def packed(path, value):
    path.write_bytes(gzip.compress(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0))


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def request_with_retry(endpoint, params):
    for attempt in range(5):
        try:
            return request(endpoint, params)
        except (HTTPError, URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                raise
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def fetch(destination):
    if not os.environ.get("NCBI_EMAIL"):
        raise ValueError("Set NCBI_EMAIL before fresh PubMed retrieval")
    destination.mkdir(parents=True, exist_ok=False)
    requests = destination / "requests"
    requests.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    queries, all_ids = [], {}
    for spec in METHODS["queries"]:
        raw = request_with_retry("esearch.fcgi", {"term": spec["query"], "retmax": 9999, "retmode": "json"})
        (requests / f"{spec['case']}_{spec['variant']}.json").write_bytes(raw)
        ids = search_ids(json.loads(raw))
        queries.append(dict(case=spec["case"], variant=spec["variant"], pmids=ids))
        all_ids.update(dict.fromkeys(ids))
        print(f"Search {spec['case']} {spec['variant']}: {len(ids)} records", flush=True)
    ids, records = list(all_ids), []
    for start in range(0, len(ids), 200):
        xml = request_with_retry("efetch.fcgi", {"id": ",".join(ids[start:start+200]), "retmode": "xml"})
        (requests / f"efetch_{start:05d}.xml").write_bytes(xml)
        batch = normalized_response(xml)
        if len(batch) != len(ids[start:start+200]) or {r["pmid"] for r in batch} != set(ids[start:start+200]):
            raise ValueError("PubMed batch incomplete or duplicated")
        records.extend(batch)
        print(f"Fetched {len(records)}/{len(ids)} unique records", flush=True)
    packed(destination / "query_records.json.gz", records)
    packed(destination / "query_memberships.json.gz", dict(
        started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        origin="Fresh retrieval, not the original historical record sets.",
        methods_sha256=hashlib.sha256((DATA / "methods.json").read_bytes()).hexdigest(), queries=queries))


def selected_data(directory):
    records = unpack(directory / "query_records.json.gz")
    memberships = unpack(directory / "query_memberships.json.gz")
    if memberships["methods_sha256"] != hashlib.sha256((DATA / "methods.json").read_bytes()).hexdigest():
        raise ValueError("Query method definitions differ from the retrieval manifest")
    by_id = {r["pmid"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate PMID in unique-record archive")
    queries = memberships["queries"]
    expected = [(q["case"], q["variant"]) for q in METHODS["queries"]]
    if [(q["case"], q["variant"]) for q in queries] != expected:
        raise ValueError("Missing, reordered or unexpected query variants")
    if set(by_id) != {pmid for q in queries for pmid in q["pmids"]}:
        raise ValueError("Query membership and record archive do not align")
    selections, union = [], {}
    for q, spec in zip(queries, METHODS["queries"]):
        if len(q["pmids"]) != len(set(q["pmids"])):
            raise ValueError("Duplicate PMID within a query")
        if spec["mesh_major"] or spec["journal_whitelist"] or {p.lower() for p in spec["exclude_publication_types"]} != EXCLUDE:
            raise ValueError("Unsupported filter difference; inspect source configuration")
        selected = [r for _, r in select_records([by_id[p] for p in q["pmids"]])]
        selections.append((q, selected))
        union.update((r["pmid"], r) for r in selected)
    return selections, [union[p] for p in sorted(union, key=int)], memberships


def audit_retrieval(directory):
    """Optional raw-response check; XML is kept outside the minimal code package."""
    _, _, manifest = selected_data(directory)
    for query in manifest["queries"]:
        path = directory / "requests" / f"{query['case']}_{query['variant']}.json"
        if search_ids(json.loads(path.read_text())) != query["pmids"]:
            raise ValueError("Raw search reply differs from saved query membership")
    paths = sorted((directory / "requests").glob("efetch_*.xml"))
    normalized = [r for path in paths for r in normalized_response(path.read_bytes())]
    if normalized != unpack(directory / "query_records.json.gz"):
        raise ValueError("Raw XML does not reconstruct the normalized record archive")
    save_csv("query_retrieval_validation.csv", [dict(queries=len(manifest["queries"]),
             xml_batches=len(paths), unique_records=len(normalized),
             excluded_book_records=sum(r.get("record_type") == "PubmedBookArticle" for r in normalized),
             status="PASS")])
    print("PASS: raw search replies and XML reconstruct every archived query record", flush=True)


def score(directory, verify=False):
    _, records, _ = selected_data(directory)
    if not verify and (directory / "query_cosines.csv").exists():
        raise FileExistsError("Saved query scores already exist")
    qtok, qmod, atok, amod = load_medcpt()
    anchors = encode_mean(qtok, qmod, [c["anchor_text"] for c in CASES])
    scores = []
    for start in range(0, len(records), 128):
        batch = records[start:start+128]
        vectors = encode_mean(atok, amod, [text_of(r) for r in batch])
        values = np.einsum("ij,kj->ik", vectors.astype(float), anchors.astype(float))
        for r, cosines in zip(batch, values):
            scores.append(dict(pmid=r["pmid"], text_sha256=hashlib.sha256(text_of(r).encode()).hexdigest(),
                               **{c["case"]: float(v) for c, v in zip(CASES, cosines)}))
        print(f"Encoded {min(start+128,len(records))}/{len(records)} unique records", flush=True)
    if verify:
        with (directory / "query_cosines.csv").open() as handle:
            archived = list(csv.DictReader(handle))
        if len(archived) != len(scores):
            raise ValueError("Rebuilt score row count differs")
        max_error = 0.0
        for old, new in zip(archived, scores):
            if any(old[k] != new[k] for k in ("pmid", "text_sha256")):
                raise ValueError("Rebuilt score ID/text mismatch")
            a = np.array([float(old[c["case"]]) for c in CASES])
            b = np.array([new[c["case"]] for c in CASES])
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
            max_error = max(max_error, float(np.max(np.abs(a-b))))
        save_csv("query_reconstruction.csv", [dict(records=len(scores), anchors=len(CASES),
                 max_cosine_abs_error=max_error, max_allowed_cosine_error=1e-5, status="PASS")])
        audit(directory, score_rows=scores)
        print("PASS: every fresh-query cosine reconstructed from archived text", flush=True)
    else:
        with (directory / "query_cosines.csv").open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(scores[0]))
            writer.writeheader(); writer.writerows(scores)


def audit(directory, score_rows=None):
    selections, records, manifest = selected_data(directory)
    if score_rows is None:
        with (directory / "query_cosines.csv").open() as handle:
            saved = list(csv.DictReader(handle))
    else:
        saved = score_rows
    scores = {r["pmid"]: r for r in saved}
    if len(scores) != len(saved) or set(scores) != {r["pmid"] for r in records}:
        raise ValueError("Saved scores do not match selected PMIDs")
    for r in records:
        s = scores[r["pmid"]]
        if s["text_sha256"] != hashlib.sha256(text_of(r).encode()).hexdigest():
            raise ValueError("Saved score text hash mismatch")
        values = np.array([float(s[c["case"]]) for c in CASES])
        if not np.isfinite(values).all() or (np.abs(values) > 1+1e-5).any():
            raise ValueError("Nonfinite or out-of-range query cosine")
    results, annual = [], []
    for q, selected in selections:
        rows = yearly_series([int(r["year"]) for r in selected], [float(scores[r["pmid"]][q["case"]]) for r in selected])
        tau, p, _ = trend(rows, "mean")
        results.append(dict(case=q["case"], variant=q["variant"], raw_records=len(q["pmids"]),
                            analyzed_records=sum(r["n"] for r in rows), n_years=len(rows),
                            tau=tau, nominal_p=p, significant_at_0_05=int(p<0.05),
                            original_direction_rule="negative" if tau < -0.05 else "positive" if tau > 0.05 else "flat",
                            raw_pmid_set_sha256=hashlib.sha256("\n".join(sorted(q["pmids"], key=int)).encode()).hexdigest()))
        annual.extend(dict(case=q["case"], variant=q["variant"], **r) for r in rows)
        print(f"{q['case']} {q['variant']}: tau={tau:+.4f}, nominal p={p:.6g}", flush=True)
    save_csv("query_sensitivity.csv", results)
    save_csv("query_sensitivity_annual.csv", annual)
    print(f"Fresh retrieval completed {manifest['completed_at_utc']}; NOT the historical retrieval.", flush=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "audit"
    directory = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DATA
    if command == "fetch":
        fetch(directory)
    elif command == "score":
        score(directory)
    elif command == "rebuild":
        score(directory, verify=True)
    elif command == "audit":
        audit(directory)
    elif command == "audit-retrieval":
        audit_retrieval(directory)
    else:
        raise SystemExit(__doc__)
