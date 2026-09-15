# Sensitivity inputs

Only the three fixed PJI, ACS and DRF anchors are analyzed.

## Evidence weighting

`methods.json` defines exact publication-type weights. Each record receives the
maximum matching weight, with a floor of 1.0. `analysis/sensitivity.py` calculates
weighted annual means using the primary anchors, 2000–2024 window and five-record
annual floor. Kendall τ is −0.7794 (PJI), −0.0600 (ACS) and +0.4275 (DRF).
Weights are a sensitivity choice, not a validated assessment of study quality.

## Retrieval sensitivity

The fifteen query definitions in `methods.json` were applied in a separately
dated retrieval on 31 August 2026; this is not reconstruction of the missing
original alternate search results. Start/end timestamps and exact PMID sets are
in `query_memberships.json.gz`. The separately retained, nonpublic
`query_records.json.gz` stores normalized records once per PMID; restore it for
text-dependent checks. Of 12,985 unique retrieved IDs, 8,806 selected records contain
8,803 distinct texts. Book entries remain in retrieval accounting but are not
encoded as articles. Annual analysis retains 2000–2024 and at least five records.

`query_cosines.csv` stores text hashes and cosines to all three primary anchors.
Full reproduction re-encodes every selected text, checks each score to
absolute tolerance 1e-5, and reruns query trends from those reconstructed scores.
The membership manifest records both the original configuration-file hash and
the release configuration hash; only development provenance fields were removed.
Query definitions, weights, record text, memberships and scores are unchanged.

ACS labels `fasciotomy_methods` and `alt_diagnosis` have identical queries and
PMID sets. Other variants alter population or topic, such as the younger-operative
DRF query scored against the older-adult anchor. Overlapping query sets are not
independent replications. Direction labels use thresholds −0.05/+0.05 and are
separate from nominal, unadjusted Kendall p-values.

`legacy_drf_anchor.npy` is the shorter DRF anchor used by the manuscript's
paraphrase/reversed-claim probes. Its text is in `methods.json`; it does not
replace the primary DRF anchor. Full reproduction verifies it from text.

For optional new searches, `analysis/query_sensitivity.py fetch NEW_DIRECTORY`
archives XML and search replies; `score`, `audit` and `audit-retrieval` check those
new inputs. New results must not silently replace frozen publication inputs.
Third-party abstract rights apply to all retained text.
