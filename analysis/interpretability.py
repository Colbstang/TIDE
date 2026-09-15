"""PJI token-contribution trajectories and centroid PCA (Figures 3–4).

Run words|wordfig|geometry|all. Contributions are additive accounting, not
causal attribution; PCA axes and their clinical labels are descriptive.
"""
import os, sys, csv
from collections import defaultdict
import numpy as np
import torch
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from common import EMB, OUT, load_medcpt, nrm, BLUE_FAMILY, ORANGE_FAMILY, DIR_BLUE, DIR_ORANGE
from source import primary_analysis_records


# ----------------------------------------------------------------------------
# words - per-term contribution TRAJECTORIES over time (interpretability centerpiece)
# ----------------------------------------------------------------------------
WORD_CONF = {
 "PJI": dict(y0=2008,
   anc="Chronic periprosthetic joint infection should be treated with two-stage exchange arthroplasty.",
   terms=["two-stage", "single-stage", "one-stage", "debridement", "DAIR", "spacer", "reimplantation", "retention", "antibiotic", "irrigation", "exchange", "reinfection"]),
}
WORD_CAP = 40


def run_words(case="PJI"):
    np.random.seed(0)
    c = WORD_CONF[case]; qtok, qmod, atok, amod = load_medcpt()
    e = qtok([c["anc"]], padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        h = qmod(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1).float()
    A = ((h * m).sum(1) / m.sum(1))[0].numpy(); A = A / np.linalg.norm(A)
    tterms = {t: atok(t, add_special_tokens=False)["input_ids"] for t in c["terms"]}
    def fa(ids, sub): return [i for i in range(len(ids) - len(sub) + 1) if ids[i:i + len(sub)] == sub]
    peryr = defaultdict(list)
    for y, txt in primary_analysis_records(case):
        if y >= c["y0"]:
            peryr[y].append(txt)
    rows = {}
    for y in sorted(peryr):
        docs = peryr[y]
        if len(docs) < 5: continue
        idx = np.random.permutation(len(docs))[:WORD_CAP]
        acc = defaultdict(float)
        for j in idx:
            enc = atok(docs[j], truncation=True, max_length=400, return_tensors="pt"); ids = enc["input_ids"][0].tolist()
            with torch.no_grad(): H = enc and amod(**enc).last_hidden_state[0].numpy()
            denom = len(H) * (np.linalg.norm(H.mean(0)) + 1e-9)
            contrib = np.einsum("ij,j->i", H.astype(float), A.astype(float)) / denom
            # Verify the exact decomposition before retaining selected term contributions.
            cosine = np.einsum("j,j->", A.astype(float), nrm(H.mean(0)).astype(float))
            np.testing.assert_allclose(contrib.sum(), cosine, atol=1e-5, rtol=0)
            for t, tid in tterms.items():
                pos = fa(ids, tid)
                acc[t] += sum(float(contrib[p:p + len(tid)].sum()) for p in pos)
        rows[y] = {t: acc[t] / len(idx) for t in c["terms"]}
    ys = sorted(rows)
    with open(f"{OUT}/word_traj_{case}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["year"] + c["terms"])
        for y in ys: w.writerow([y] + [round(rows[y][t], 6) for t in c["terms"]])
    print(f"=== {case}: per-term contribution to anchor similarity, early(first5) -> late(last5) + trend ===")
    out = []
    for t in c["terms"]:
        v = [rows[y][t] for y in ys]; early = np.mean(v[:5]); late = np.mean(v[-5:]); tau = kendalltau(ys, v)[0]
        out.append((t, early, late, late - early, tau))
    for t, e_, l_, dl, tau in sorted(out, key=lambda x: x[3]):
        arrow = "UP  " if dl > 0 else "DOWN"
        print(f"  {t:13s} early={e_:+.4f} late={l_:+.4f}  Δ={dl:+.4f} {arrow} tau={tau:+.2f}")
    print(f"\nwrote out/word_traj_{case}.csv  ({len(ys)} yrs)")


# -- wordfig - named trajectory figure from the csv (no MedCPT) ----------------
# Palette carries the case scheme forward: receding (declining) terms = blue family,
# rising terms = orange family, flat/null terms = grey family.
WORD_SHOW = {
 "PJI": dict(terms=[
   ("two-stage", BLUE_FAMILY[0], True), ("debridement", BLUE_FAMILY[1], False), ("reimplantation", BLUE_FAMILY[2], False),
   ("DAIR", ORANGE_FAMILY[0], True), ("one-stage", ORANGE_FAMILY[1], False), ("single-stage", ORANGE_FAMILY[2], False)]),
}


def _smooth(v, k=3):
    v = np.asarray(v, float)
    if len(v) < k: return v
    pad = np.r_[v[:k // 2][::-1], v, v[-(k // 2):][::-1]]
    return np.convolve(pad, np.ones(k) / k, mode="valid")[:len(v)]


def run_wordfig(case="PJI"):
    cfg = WORD_SHOW[case]
    rows = list(csv.DictReader(open(f"{OUT}/word_traj_{case}.csv")))
    years = np.array([int(r["year"]) for r in rows])
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for term, color, emph in cfg["terms"]:
        if term not in rows[0]:
            continue
        y = _smooth([float(r[term]) for r in rows])
        ax.plot(years, y, color=color, lw=3.4 if emph else 2.0, alpha=0.97 if emph else 0.85,
                zorder=3 if emph else 2, label=term)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("year", fontsize=12); ax.set_ylabel("contribution to anchor similarity", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # whole-number years
    ax.grid(alpha=0.22)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=11, frameon=False)  # term legend, outside right
    fig.savefig(f"{OUT}/fig_word_trajectories_{case}.png", dpi=600, bbox_inches="tight")
    print(f"wrote out/fig_word_trajectories_{case}.png")


# ----------------------------------------------------------------------------
# geometry - PJI per-year-centroid PCA probe + Figure 4 (exploratory, no stats)
# ----------------------------------------------------------------------------
def run_geometry():
    qtok, qmod, atok, amod = load_medcpt()
    def emb(term):
        e = atok(term, return_tensors="pt")
        with torch.no_grad(): H = amod(**e).last_hidden_state[0].numpy()
        return H[1:-1].mean(0)  # mean of content tokens (excl CLS/SEP)
    EMBP = f"{EMB}/trial_pji_two_stage_exchange"
    av = np.load(f"{EMBP}/abstract_vectors.npy"); yr = np.load(f"{EMBP}/years.npy")
    m = (yr >= 2008) & (yr <= 2024); av, yr = av[m], yr[m].astype(int)
    years = [y for y in np.unique(yr) if (yr == y).sum() >= 5]
    yc = np.array([av[yr == y].mean(0) for y in years])  # per-year centroids (raw mean-pooled)

    # PCA of per-year centroids + name poles by clinical-term loadings
    VOCAB = ["two-stage", "single-stage", "one-stage", "DAIR", "debridement", "reimplantation", "spacer",
             "antibiotic", "culture", "synovial", "diagnosis", "biomarker", "sequencing", "infection",
             "arthroplasty", "revision", "outcome", "complication", "reinfection", "irrigation", "sinus tract"]
    ve = {t: emb(t) for t in VOCAB}
    pca = PCA(n_components=3, svd_solver="full").fit(yc.astype(float))   # exact solver -> deterministic
    print(f"\n=== PCA of per-year centroids - variance explained: "
          f"PC1 {pca.explained_variance_ratio_[0]:.0%}, PC2 {pca.explained_variance_ratio_[1]:.0%}, PC3 {pca.explained_variance_ratio_[2]:.0%} ===")
    pc1_scores = np.einsum("ij,j->i", yc.astype(float) - pca.mean_, pca.components_[0])
    ktau, kp = kendalltau(years, pc1_scores)
    pd.DataFrame([dict(case="PJI", n_years=len(years), pc1_variance=round(float(pca.explained_variance_ratio_[0]), 6),
                      pc2_variance=round(float(pca.explained_variance_ratio_[1]), 6),
                      pc3_variance=round(float(pca.explained_variance_ratio_[2]), 6),
                      year_pc1_tau=round(float(ktau), 6), nominal_p=f"{kp:.6g}")]).to_csv(
                          f"{OUT}/geometry_summary.csv", index=False)
    print(f"  year vs PC1 score: Kendall tau = {ktau:+.3f}, p = {kp:.1e}  (monotone drift along PC1)")
    for k in range(2):
        load = {t: float(np.einsum("j,j->", ve[t].astype(float) - pca.mean_, pca.components_[k]))
                for t in VOCAB}
        s = sorted(load.items(), key=lambda x: x[1])
        neg = ", ".join(f"{t}" for t, _ in s[:4]); pos = ", ".join(f"{t}" for t, _ in s[-4:])
        yk = np.einsum("ij,j->i", yc.astype(float) - pca.mean_, pca.components_[k])
        drift = "EARLY->LATE moves toward [" + ("+pole" if yk[-5:].mean() > yk[:5].mean() else "-pole") + "]"
        print(f"  PC{k+1}:  -pole {{{neg}}}   <->   +pole {{{pos}}}")
        print(f"         year trajectory: early {yk[:5].mean():+.3f} -> late {yk[-5:].mean():+.3f}   ({drift})")
    np.save(f"{OUT}/pji_year_centroids.npy", yc)
    print("\nwrote out/pji_year_centroids.npy")

    _geometry_figure()


def _geometry_figure():
    yc = np.load(f"{OUT}/pji_year_centroids.npy")
    years = list(range(2008, 2008 + len(yc)))  # 2008..2024 (kept years)
    P = PCA(n_components=2, svd_solver="full").fit(yc.astype(float))
    Z = np.einsum("ij,kj->ik", yc.astype(float) - P.mean_, P.components_)
    # Orient later years to the right; clinical pole labels are post hoc.
    if Z[-5:, 0].mean() < Z[:5, 0].mean(): Z[:, 0] *= -1
    blor = LinearSegmentedColormap.from_list("blor", [DIR_BLUE, "#f2f2f2", DIR_ORANGE])  # early=blue -> late=orange
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(Z[:, 0], Z[:, 1], "-", color="#999", lw=1, alpha=0.6, zorder=1)
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=years, cmap=blor, s=110, zorder=3, edgecolor="white", lw=0.7)
    for i, y in enumerate(years):
        if y % 4 == 0 or y in (years[0], years[-1]):
            ax.annotate(str(y), (Z[i, 0], Z[i, 1]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    fig.colorbar(sc, ax=ax, label="year", fraction=0.04, pad=0.02)
    ax.set_xlabel(f"PC1  ({P.explained_variance_ratio_[0]:.0%} var):   diagnosis / microbiology  ⟷  treatment procedure", fontsize=11)
    ax.set_ylabel(f"PC2  ({P.explained_variance_ratio_[1]:.0%} var):   lab methods  ⟷  outcomes / complications", fontsize=11)
    ax.tick_params(labelsize=10)
    xl = ax.get_xlim()
    ax.text(xl[0], ax.get_ylim()[1], "  ← culture · diagnosis · sequencing", fontsize=9, color="#444", va="top")
    ax.text(xl[1], ax.get_ylim()[1], "spacer · DAIR · revision →  ", fontsize=9, color="#444", va="top", ha="right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_geometry_pji.png", dpi=600)
    print("wrote out/fig_geometry_pji.png (blue->orange year ramp)")



def run_all():
    run_words("PJI")
    run_wordfig("PJI")
    run_geometry()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"all": run_all, "words": run_words, "wordfig": run_wordfig, "geometry": run_geometry}[command]()
