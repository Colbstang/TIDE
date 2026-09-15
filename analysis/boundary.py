"""Sampled context probes and two reversed-claim sentence pairs (Table 4)."""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, OUT, load_medcpt, nrm


# ----------------------------------------------------------------------------
# Contextual token representations and sentence-level minimal pairs.
# ----------------------------------------------------------------------------
ANCHOR = {
    "PJI": "Chronic periprosthetic joint infection should be treated with two-stage exchange arthroplasty.",
    "ACS": "Acute compartment syndrome should be treated with emergent fasciotomy.",
    "DRF": "Distal radius fractures in patients older than 65 years should be managed nonoperatively.",
}
RAWF = {"PJI": "trial_pji_two_stage_exchange", "ACS": "trial_acute_compartment_fasciotomy",
        "DRF": "trial_aaos_distal_radius_elderly_nonop"}
KEYTERMS = {
    "PJI": ["two-stage", "single-stage", "debridement", "exchange"],
    "ACS": ["fasciotomy", "decompression", "monitoring"],
    "DRF": ["nonoperative", "operative", "casting", "fixation"],
}
# minimal pairs: (concordant-with-anchor sentence, discordant sentence) - SAME multiset of words
PAIRS = {
    "DRF": [("Nonoperative management was superior to operative fixation in elderly distal radius fractures.",
             "Operative fixation was superior to nonoperative management in elderly distal radius fractures.")],
    "PJI": [("Two-stage exchange was superior to single-stage exchange for periprosthetic joint infection.",
             "Single-stage exchange was superior to two-stage exchange for periprosthetic joint infection.")],
}

_M = {}   # filled with the encoders at run time


def art_mean(text):
    import torch
    atok, amod = _M["atok"], _M["amod"]
    e = atok(text, truncation=True, max_length=400, return_tensors="pt")
    with torch.no_grad():
        H = amod(**e).last_hidden_state[0]
    return nrm(H.mean(0).numpy())


def qry_mean(text):
    import torch
    qtok, qmod = _M["qtok"], _M["qmod"]
    e = qtok([text], padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        h = qmod(**e).last_hidden_state
    m = e["attention_mask"].unsqueeze(-1).float()
    return nrm(((h * m).sum(1) / m.sum(1))[0].numpy())


def term_isolated(term):
    """embed the term ALONE; mean of its content-token vectors (no surrounding context)."""
    import torch
    atok, amod = _M["atok"], _M["amod"]
    tids = atok(term, add_special_tokens=False)["input_ids"]
    e = atok(term, return_tensors="pt")
    with torch.no_grad():
        H = amod(**e).last_hidden_state[0].numpy()
    ids = e["input_ids"][0].tolist()
    for i in range(len(ids) - len(tids) + 1):
        if ids[i:i + len(tids)] == tids:
            return nrm(H[i:i + len(tids)].mean(0)), tids
    return nrm(H[1:1 + len(tids)].mean(0)), tids


def term_in_context(docs, tids, cap=200):
    """Stop after the document reaching cap; two hits/doc can yield cap + 1."""
    import torch
    atok, amod = _M["atok"], _M["amod"]
    vecs = []
    for txt in docs:
        enc = atok(txt, truncation=True, max_length=400, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        pos = [i for i in range(len(ids) - len(tids) + 1) if ids[i:i + len(tids)] == tids]
        if not pos:
            continue
        with torch.no_grad():
            H = amod(**enc).last_hidden_state[0].numpy()
        for p in pos[:2]:
            vecs.append(nrm(H[p:p + len(tids)].mean(0)))
        if len(vecs) >= cap:
            break
    return vecs


def run_attention():
    np.random.seed(0)
    _M["qtok"], _M["qmod"], _M["atok"], _M["amod"] = load_medcpt()
    rows = []
    pairrows = []

    print("=" * 78)
    print("(A) ISOLATED vs IN-CONTEXT - how far context moves each key term,")
    print("    and whether context pulls it TOWARD the recommendation anchor")
    print("=" * 78)
    corpus = {c: json.load(open(f"{RAW}/{RAWF[c]}_pubmed_2000_2024.json")) for c in ANCHOR}
    for c in ["PJI", "ACS", "DRF"]:
        anc = qry_mean(ANCHOR[c])
        docs_all = [(r.get("title", "") + ". " + (r.get("abstract") or "")) for r in corpus[c]]
        np.random.shuffle(docs_all)
        print(f"\n  {c}  (anchor: \"{ANCHOR[c][:60]}...\")")
        print(f"    {'term':14} {'displace':>9} {'cos(iso,anc)':>12} {'cos(ctx,anc)':>12} {'pull→anchor':>11}")
        for term in KEYTERMS[c]:
            iso, tids = term_isolated(term)
            vecs = term_in_context(docs_all, tids)
            if len(vecs) < 10:
                print(f"    {term:14} (only {len(vecs)} occ - skip)")
                continue
            ctx = nrm(np.mean(vecs, 0))
            disp = 1 - float(iso @ ctx)
            ci, cc = float(iso @ anc), float(ctx @ anc)
            pull = cc - ci
            print(f"    {term:14} {disp:>9.3f} {ci:>12.3f} {cc:>12.3f} {pull:>+11.3f}")
            rows.append(dict(case=c, vignette="A_displacement", term=term, displacement=round(disp, 3),
                             cos_iso_anchor=round(ci, 3), cos_ctx_anchor=round(cc, 3), pull_to_anchor=round(pull, 3)))

    print("\n" + "=" * 78)
    print("(B) MINIMAL-PAIR TEST - same bag of words, flipped meaning.")
    print("    TF-IDF is order-blind (identical); does MedCPT attention separate them,")
    print("    in the recommendation-concordant direction?")
    print("=" * 78)
    for c, pairs in PAIRS.items():
        anc_q = qry_mean(ANCHOR[c])
        for concordant, discordant in pairs:
            mc = float(art_mean(concordant) @ anc_q)
            md = float(art_mean(discordant) @ anc_q)
            vec = TfidfVectorizer(stop_words="english")
            X = vec.fit_transform([ANCHOR[c], concordant, discordant])
            tc = float(cosine_similarity(X[0], X[1])[0, 0])
            td = float(cosine_similarity(X[0], X[2])[0, 0])
            print(f"\n  {c}:")
            print(f"    concordant : \"{concordant}\"")
            print(f"    discordant : \"{discordant}\"")
            print(f"    TF-IDF  cos->anchor:  concordant={tc:.4f}  discordant={td:.4f}  diff={tc-td:+.4f}  (≈0 = blind)")
            print(f"    MedCPT  cos->anchor:  concordant={mc:.4f}  discordant={md:.4f}  diff={mc-md:+.4f}  (positive favors the concordant probe)")
            rows.append(dict(case=c, vignette="B_minpair", term="flip", displacement="",
                             cos_iso_anchor="", cos_ctx_anchor="",
                             pull_to_anchor=round(mc - md, 4)))
            rows.append(dict(case=c, vignette="B_minpair_tfidf", term="flip", displacement="",
                             cos_iso_anchor="", cos_ctx_anchor="", pull_to_anchor=round(tc - td, 4)))
            for label, sentence, medcpt, tfidf in [("concordant", concordant, mc, tc),
                                                  ("discordant", discordant, md, td)]:
                pairrows.append(dict(case=c, direction=label, anchor=ANCHOR[c], sentence=sentence,
                                     medcpt_cosine=round(medcpt, 6), tfidf_cosine=round(tfidf, 6)))

    pd.DataFrame(rows).to_csv(f"{OUT}/attention_reshaping.csv", index=False)
    pd.DataFrame(pairrows).to_csv(f"{OUT}/minimal_pairs.csv", index=False)
    print("\nwrote out/attention_reshaping.csv and out/minimal_pairs.csv")


SUBS = {"attention": run_attention}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "attention"
    for name in (SUBS if which == "all" else [which]):
        print(f"\n########## boundary.py :: {name} ##########")
        SUBS[name]()
