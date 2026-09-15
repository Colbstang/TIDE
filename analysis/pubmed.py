"""Optional fresh retrieval of the three configured queries (NOT exact reproduction).

Usage: python analysis/pubmed.py NEW_DIRECTORY
Never overwrites the archive. Normalization follows the historical PubMed client;
requests/results are retained together so a future search has explicit provenance.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def normalize_article(article):
    """Keep original JournalIssue year precedence, including MedlineDate fallback."""
    def text(path):
        node = article.find(path)
        return "" if node is None else "".join(node.itertext()).strip()

    year_text = article.findtext(".//JournalIssue/PubDate/Year")
    if year_text is None:
        date = article.findtext(".//JournalIssue/PubDate/MedlineDate") or ""
        year_text = "".join(c for c in date if c.isdigit())[:4]
    try:
        year = int(year_text)
    except (TypeError, ValueError):
        year = None
    return dict(
        pmid=text(".//PMID"), title=text(".//ArticleTitle"),
        abstract=" ".join("".join(n.itertext()).strip() for n in article.findall(".//AbstractText")
                          if "".join(n.itertext()).strip()),
        journal=text(".//Journal/Title"), year=year,
        mesh_major=[(n.text or "").strip() for n in article.findall(".//MeshHeading/DescriptorName")
                    if n.attrib.get("MajorTopicYN") == "Y" and (n.text or "").strip()],
        publication_types=[(n.text or "").strip() for n in article.findall(".//PublicationTypeList/PublicationType")
                           if (n.text or "").strip()],
        languages=[(n.text or "").strip() for n in article.findall(".//Language") if (n.text or "").strip()],
    )


def normalize_response(xml):
    """Normalize every returned PMID while keeping book entries excluded explicitly."""
    records = []
    for item in ET.fromstring(xml):
        if item.tag == "PubmedArticle":
            records.append(normalize_article(item))
        elif item.tag == "PubmedBookArticle":
            pmid = item.findtext("BookDocument/PMID")
            if not pmid:
                raise ValueError("PubMed book response missing PMID")
            records.append(dict(
                pmid=pmid,
                record_type="PubmedBookArticle",
                exclusion="Not a PubmedArticle; excluded by original ingestion rule",
            ))
        else:
            raise ValueError(f"Unexpected PubMed response element: {item.tag}")
    return records


def request(endpoint, params):
    # No credentials required; one serial request per >=0.34 s (below 3/s).
    time.sleep(0.34)
    url = BASE + endpoint + "?" + urlencode({"db": "pubmed", "tool": "TIDE", "email": os.environ["NCBI_EMAIL"], **params})
    with urlopen(url, timeout=60) as response:
        return response.read()


def search_ids(payload):
    result = payload["esearchresult"]
    if "error" in result or payload.get("error") or result.get("errorlist"):
        raise ValueError(f"PubMed search error: {result}")
    ids = [str(v) for v in result["idlist"]]
    if len(ids) != int(result["count"]) or len(set(ids)) != len(ids):
        raise ValueError("Search truncated or duplicated; do not use a partial retrieval")
    if not ids:
        raise ValueError("Search returned no records")
    return ids


def fetch(destination):
    if not os.environ.get("NCBI_EMAIL"):
        raise ValueError("Set NCBI_EMAIL to your contact email before a fresh PubMed retrieval")
    destination.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "anchors.json").read_text())
    for case in config["cases"]:
        folder = destination / case["case"]
        folder.mkdir()
        query = case["pubmed_query"]
        payload = request("esearch.fcgi", {"term": query, "retmax": 9999, "retmode": "json"})
        (folder / "esearch.json").write_bytes(payload)
        ids = search_ids(json.loads(payload))
        records = []
        for start in range(0, len(ids), 200):
            xml = request("efetch.fcgi", {"id": ",".join(ids[start:start + 200]), "retmode": "xml"})
            (folder / f"efetch_{start:05d}.xml").write_bytes(xml)
            records.extend(normalize_response(xml))
        if len(records) != len(ids) or {r["pmid"] for r in records} != set(ids):
            raise ValueError(f"{case['case']}: missing or duplicated fetched records")
        (folder / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
        (folder / "retrieval.json").write_text(json.dumps(dict(query=query, pmids=ids,
            retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
            notice="Fresh search; not the historical reproduction input."), indent=2) + "\n")
        print(f"{case['case']}: {len(records)} freshly retrieved records", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    fetch(Path(sys.argv[1]).resolve())
