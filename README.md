# TIDE: Temporal Index of Divergent Evidence

A reproducible **three-case methods proof of concept** in orthopedic surgery.
TIDE measures topical alignment between a fixed recommendation anchor and its
year-binned literature. It does not determine evidentiary support, treatment
superiority, guideline validity, or clinical screening accuracy.

| Case | Analyzed records | Included years | Kendall τ |
|---|---:|---|---:|
| Periprosthetic joint infection (PJI) | 2,405 | 2008–2024 | −0.838 |
| Acute compartment syndrome (ACS) | 1,463 | 2000–2024 | −0.067 |
| Distal radius fracture (DRF) | 636 | 2000–2024, except 2001 | +0.420 |

These are 4,504 case-record observations and 4,503 distinct PubMed IDs. The cases
were selected for expected trajectories, not as an unbiased validation sample.

## Reproduce

Use Python 3.13. The public checkout contains queries, PMID/year mappings,
text hashes, frozen vectors and numerical results—not article titles or abstracts.
No model download or live PubMed search is needed for the default check.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python reproduce.py
```

This verifies 48 public input/reference checksums, runs public-release tests,
and reproduces the three primary trends and vector-only robustness checks
(seven numeric tables byte-for-byte), and regenerates Figures 1–2. It does **not** recompute
text-dependent lexical, weighting, query-selection or token analyses.

For all core analyses, restore the separate frozen text archive at the four
paths listed in `reproduce.py:PRIVATE_TEXT`, then run:

```bash
python reproduce.py --with-text --output verification-core
```

The complete frozen text archive is retained by the corresponding author;
access is subject to applicable rights and permissions. Contact
Cgrames@students.llu.edu to discuss access. The code checks its original hashes.
Fresh retrieval by the public PMIDs may return changed text and is not an exact
replacement. For **archived text → embeddings → all manuscript analyses and four
figures**, after restoring the same archive:

```bash
python -m pip install -r requirements.txt
python reproduce.py --full --output verification
```

On Linux, first install CPU-only PyTorch:
`python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu`.
The full run is CPU-intensive and downloads two revision-pinned MedCPT models
unless cached. Set `TIDE_THREADS=4` to limit threads; `--offline` requires an
existing model cache. GitHub checks the public vector-only subset on Linux/macOS;
text-dependent and full reconstruction run locally with the separate archive.
[Workflow results](https://github.com/Colbstang/TIDE/actions)

`--output NEW_DIRECTORY` retains verification evidence and its exact scope.
`--package NEW_FILE.zip` creates a deterministic public code/vector/result archive
after verification; it always excludes article text, even when restored locally.
Reproduction never overwrites the frozen inputs or reference outputs.

## Manuscript map

Released analyses support the manuscript's tables, figures, comparisons and
stated method checks. There is no application layer or development notebook.

| Manuscript content | Implementation | Main outputs in `out/` |
|---|---|---|
| Table 1, counts and selection | `source.py` | `records.csv`, `source_counts.csv`, `annual_series.csv` |
| Table 2, trends and sensitivity | `drift.py`, `robustness.py`, `sensitivity.py`, `source.py` | `case_study_metrics.csv`, `drift_resampling.csv`, `robustness_*.csv`, `evidence_weighting.csv`, `matched_cls.csv` |
| Table 3, publication and evidence-volume baselines | `source.py` | `matched_baselines.csv` |
| Paper-level and annual-document lexical comparisons | `comparisons.py` | `tfidf_variants.csv`, `tfidf_annual.csv` |
| Retrieval and cross-anchor checks | `query_sensitivity.py`, `comparisons.py` | `query_sensitivity.csv`, `cross_anchors.csv` |
| Table 4, reversed-meaning probes; isolated/in-context analysis | `boundary.py` | `minimal_pairs.csv`, `attention_reshaping.csv` |
| Figures 1–2 | `drift.py` | `fig_pipeline.png`, `fig_drift_curves.png` |
| Figures 3–4 | `interpretability.py` | `word_traj_PJI.csv`, `geometry_summary.csv`, `fig_word_trajectories_PJI.png`, `fig_geometry_pji.png` |

Scripts are in `analysis/`. [paper_core.csv](out/paper_core.csv) and
[paper_probes.csv](out/paper_probes.csv) are generated numeric indexes with
source-table/column pointers. Annual values and plotted coordinates remain in
their detailed tables. External literature statistics and author-supplied
assessments are not TIDE-generated outputs.

## Method boundaries

- Exact anchors, queries and exclusions: `anchors.json`. Primary inputs are
  title plus available abstract, joined by one space; 29 analyzed records have
  titles only. English/publication-type filtering precedes embedding. Analysis
  retains 2000–2024 and years with at least five records.
- Primary encoding uses 512-token truncation, attention-mask mean pooling
  including special tokens, and L2 normalization. Matched [CLS] changes pooling
  on the same inputs; it is not the entire native MedCPT retrieval pipeline.
- Central 95% within-year resampling ranges describe conditional perturbations,
  not calibrated confidence intervals. They can exclude the observed τ. Range,
  sign, year-shuffle and phase checks use 1,500, 1,000, 1,000 and 5,000 replicates,
  respectively, with seed 0. Year shuffling precedes reapplying the annual floor;
  the phase null treats retained bins as equally spaced, including DRF's gap.
- Post hoc TF-IDF comparisons retain all four combinations of annual-document
  versus paper-level fitting/scoring. Paper-level fitting/scoring recovers the
  qualitative three-case pattern; embedding superiority is not established.
  DRF rises toward all three anchors, limiting recommendation specificity.
- Figure 3 samples up to 40 primary-eligible PJI records/year (seed 0), uses 400
  tokens, and displays six of twelve recorded terms with centered three-bin
  smoothing. Selected contributions do not exhaust the score. Figure 4 is
  descriptive PCA of annual centroids; clinical axis labels are post hoc.
- Isolated/in-context probes shuffle raw records (seed 0) and retain at most two
  occurrences per document, stopping after the document that reaches 200
  occurrences (thus at most 201). They use 400-token article inputs and 64-token
  query inputs, unlike the primary analysis. DRF reversed-claim and paraphrase probes use the shorter anchor
  in `data/sensitivity/methods.json`, not its primary anchor.
- Evidence weights and dated retrieval provenance are documented in
  [data/sensitivity/README.md](data/sensitivity/README.md).

## Data and verification

`provenance.json` pins model revisions and input/reference hashes. Primary τ must
agree to `1e-12`; rebuilt normalized vectors use absolute tolerance `1e-5` and
zero relative tolerance. Rebuilt tables use documented numeric tolerances with
stricter checks for tiny p-values. Identifiers, schemas and order must match;
nonfinite values fail. Figures are regenerated, not certified pixel-identical.

For complete reconstruction, frozen normalized records are the reproducible starting point. The original
primary search timestamp and raw XML were not archived. Optional fresh retrieval
with `NCBI_EMAIL=you@example.org python analysis/pubmed.py NEW_DIRECTORY` cannot
recreate that event and never silently replaces frozen inputs. Archived PubMed
data are a fixed snapshot, not necessarily the latest NLM records.

Code is Apache-2.0; it does not license third-party abstracts. NLM is the source
of the bibliographic data and does not endorse TIDE. See [NLM data terms](https://www.nlm.nih.gov/databases/download.html).
Article text is excluded from public Git history, packages and CI artifacts.
The separate text archive is not licensed for public redistribution by this repo.
