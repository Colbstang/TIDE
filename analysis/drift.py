"""Primary trends, matched lexical baselines, and Figures 1–2.

Run metrics|figure|pipeline|all. Each stochastic entrypoint resets seed 0.
"""
import os, sys, ast, csv
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EMB, OUT, DIR_BLUE, DIR_GREY, DIR_ORANGE
from source import CASES

Y0, Y1, MINPUB = 2000, 2024, 5


# ----------------------------------------------------------------------------
# metrics - manuscript Tables 1 & 2, reproduced from cached MedCPT vectors
#   TIDE: annual MEAN cosine, years >=5 pubs, Kendall tau primary, Theil-Sen slope.
#   Year-shuffle null; within-year bootstrap same-sign; TF-IDF / pub-count / evidence-volume baselines.
# ----------------------------------------------------------------------------
HIGH_EV = ("randomized controlled trial", "clinical trial", "comparative study", "multicenter study")


def parse_pt(v):
    """publication_types is sometimes a JSON-list string, sometimes a list."""
    if isinstance(v, list):
        return [str(x).lower() for x in v]
    if isinstance(v, str) and v.strip():
        try:
            x = ast.literal_eval(v)
            return [str(i).lower() for i in x] if isinstance(x, (list, tuple)) else [v.lower()]
        except Exception:
            return [v.lower()]
    return []


def annual_mean_cos(case):
    av = np.load(f"{EMB}/{case['d']}/abstract_vectors.npy").astype(float)
    yr = np.load(f"{EMB}/{case['d']}/years.npy")
    gv = np.load(f"{EMB}/{case['d']}/guideline_vector.npy").ravel().astype(float)
    av = av / np.clip(np.linalg.norm(av, axis=1, keepdims=True), 1e-9, None)
    gv = gv / np.linalg.norm(gv)
    # Float64 scalar reduction avoids platform-dependent float32 BLAS rounding.
    cos = np.einsum("ij,j->i", av, gv)
    if not np.isfinite(cos).all():
        raise ValueError("Nonfinite primary cosine; refusing to silently drop records")
    m = (yr >= Y0) & (yr <= Y1) & np.isfinite(cos)
    return yr[m].astype(int), cos[m]


def trend(years, vals):
    by = pd.Series(vals).groupby(years)
    g = pd.DataFrame({"mean": by.mean(), "n": by.size()}).reset_index().rename(columns={"index": "year"})
    g = g[g["n"] >= MINPUB].sort_values("year")
    tau = kendalltau(g["year"], g["mean"])[0]
    slope = theilslopes(g["mean"], g["year"])[0] * 1e4
    return g, float(tau), float(slope)


def year_shuffle_p(years, vals, tau_obs, B=1000):
    cnt = 0
    for _ in range(B):
        ys = np.random.permutation(years)
        g, t, _ = trend(ys, vals)
        if abs(t) >= abs(tau_obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (B + 1)


def boot_same_sign(g_years, years, vals, tau_obs, B=1000):
    sgn = np.sign(tau_obs); same = 0
    yv = {y: vals[years == y] for y in g_years}
    for _ in range(B):
        ry, rm = [], []
        for y in g_years:
            s = yv[y]
            ry.append(y); rm.append(np.mean(np.random.choice(s, len(s), replace=True)))
        t = kendalltau(ry, rm)[0]
        if np.sign(t) == sgn:
            same += 1
    return 100.0 * same / B


def tfidf_pubcount_evid(case, records):
    """Annual-document TF-IDF and volume baselines on the supplied matched records."""
    per_year_docs, pubcount, evid = {}, {}, {}
    for r in records:
        y = r.get("year")
        try:
            y = int(y)
        except Exception:
            continue
        if not (Y0 <= y <= Y1):
            continue
        ab = (r.get("title", "") + ". " + (r.get("abstract") or "")).strip()
        per_year_docs.setdefault(y, []).append(ab)
        pubcount[y] = pubcount.get(y, 0) + 1
        pts = parse_pt(r.get("publication_types") or r.get("pub_types") or r.get("publication_type"))
        if any(any(h in p for h in HIGH_EV) for p in pts):
            evid[y] = evid.get(y, 0) + 1
    yrs = sorted([y for y in per_year_docs if pubcount.get(y, 0) >= MINPUB])
    corpus = [" ".join(per_year_docs[y]) for y in yrs]
    vec = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)
    X = vec.fit_transform([case["text"]] + corpus)
    sims = cosine_similarity(X[0], X[1:]).ravel()
    tau_tfidf = kendalltau(yrs, sims)[0]
    tau_pub = kendalltau(yrs, [pubcount[y] for y in yrs])[0]
    tau_evid = kendalltau(yrs, [evid.get(y, 0) for y in yrs])[0]
    hi = sum(evid.get(y, 0) for y in yrs); tot = sum(pubcount[y] for y in yrs)
    return float(tau_tfidf), float(tau_pub), float(tau_evid), 100.0 * hi / max(tot, 1)


def run_metrics():
    np.random.seed(0)
    rows = []
    for case in CASES:
        name = case["case"]
        c = dict(d=case["trial_id"], text=case["anchor_text"])
        years, cos = annual_mean_cos(c)
        g, tau, slope = trend(years, cos)
        n = int(g["n"].sum()); yspan = f"{int(g['year'].min())}-{int(g['year'].max())}"
        ysp = year_shuffle_p(years, cos, tau)
        boot = boot_same_sign(list(g["year"]), years, cos, tau)
        # Preserve unrounded Monte Carlo probabilities: DRF's 5/1001 rounds to .005.
        rows.append(dict(case=name, n=n, years=yspan, tau=round(tau, 3),
                         slope_e4=round(slope, 1), yearshuffle_p=round(ysp, 3),
                         yearshuffle_exact_p=ysp, yearshuffle_reps=1000,
                         yearshuffle_exceedances=round(ysp * 1001) - 1,
                         boot_samesign_pct=round(boot, 1), bootstrap_reps=1000))
        print(f"{name}: n={n}, tau={tau:+.6f}, year-shuffle p={ysp:.8g}")
    pd.DataFrame(rows).to_csv(f"{OUT}/case_study_metrics.csv", index=False)


# ----------------------------------------------------------------------------
# Figure 2: annual mean cosine and Theil-Sen trend
#   + conditional resampling range and trend band; neither has calibrated coverage.
# ----------------------------------------------------------------------------
FIG_CASES = [("PJI", "trial_pji_two_stage_exchange", "PJI two-stage exchange", "divergence", DIR_BLUE),
             ("ACS", "trial_acute_compartment_fasciotomy", "ACS fasciotomy", "no directional trend", DIR_GREY),
             ("DRF", "trial_aaos_distal_radius_elderly_nonop", "DRF elderly nonoperative", "convergence", DIR_ORANGE)]


def fig_series(d):
    yr, cos = annual_mean_cos({"d": d})
    yv = {y: cos[yr == y] for y in np.unique(yr) if (yr == y).sum() >= 5}
    return yv


def _drift_panel(ax, name, d, title, exp, col):
    yv = fig_series(d); ys = sorted(yv); ysa = np.array(ys)
    means = np.array([yv[y].mean() for y in ys])
    ses = np.array([yv[y].std(ddof=1) / np.sqrt(len(yv[y])) for y in ys])  # SE of each yearly mean
    tau = kendalltau(ys, means)[0]; sl, ic, _, _ = theilslopes(means, ys)
    bt, lines = [], []
    for _ in range(1500):
        rm = np.array([np.random.choice(yv[y], len(yv[y]), True).mean() for y in ys])
        bt.append(kendalltau(ys, rm)[0]); s, i, _, _ = theilslopes(rm, ys); lines.append(i + s * ysa)
    lo, hi = np.percentile(bt, [2.5, 97.5]); band = np.percentile(np.array(lines), [2.5, 97.5], axis=0)
    ax.fill_between(ys, band[0], band[1], color=col, alpha=0.15, zorder=1)             # bootstrap trend band
    ax.plot(ys, ic + sl * ysa, color=col, lw=2.2, zorder=2)                            # Theil-Sen trend
    ax.errorbar(ys, means, yerr=ses, fmt="o", ms=5, color=col, ecolor=col,
                elinewidth=1, capsize=2, alpha=0.9, zorder=3)                          # yearly mean +/- 1 SE
    obs = "divergence" if tau < -0.1 else ("convergence" if tau > 0.1 else "no detectable trend")
    ax.set_title(f"{title}\nexpected: {exp}", fontsize=13)
    ax.set_xlabel("year", fontsize=12); ax.set_ylabel("mean cosine to recommendation", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    return dict(case=name, title=title, tau=tau, lo=lo, hi=hi, obs=obs, col=col)


def _legend_panel(ax, stats):
    """Bottom-right 2x2 cell: the per-case stat boxes pulled out of the plots, in larger text."""
    ax.axis("off")
    ax.text(0.0, 0.98, "Trend on yearly means (Kendall $\\tau$)", fontsize=15, fontweight="bold",
            transform=ax.transAxes, va="top")
    y = 0.80
    for s in stats:
        ax.text(0.0, y, s["title"], fontsize=14, fontweight="bold", color=s["col"], transform=ax.transAxes, va="top")
        ax.text(0.0, y - 0.075, f"$\\tau$ = {s['tau']:+.2f}    95% resampling range [{s['lo']:+.2f}, {s['hi']:+.2f}]",
                fontsize=12.5, color="black", transform=ax.transAxes, va="top")
        ax.text(0.0, y - 0.145, f"observed: {s['obs']}", fontsize=12.5, color=s["col"], transform=ax.transAxes, va="top")
        y -= 0.27


def run_figure():
    np.random.seed(0)
    cases = {name: (name, d, title, exp, col) for (name, d, title, exp, col) in FIG_CASES}
    # 2x2: PJI top-left, DRF top-right, ACS bottom-left, trend-summary legend bottom-right.
    # No in-image title/footnote - those live in the figure caption.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.93, bottom=0.08, hspace=0.42, wspace=0.26)
    stats = [
        _drift_panel(axes[0, 0], *cases["PJI"]),
        _drift_panel(axes[0, 1], *cases["DRF"]),
        _drift_panel(axes[1, 0], *cases["ACS"]),
    ]
    _legend_panel(axes[1, 1], stats)
    fig.savefig(f"{OUT}/fig_drift_curves.png", dpi=600)
    with open(f"{OUT}/drift_resampling.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "tau", "range_low", "range_high", "bootstrap_reps", "method"])
        for s in stats:
            w.writerow([s["case"], round(float(s["tau"]), 6), round(float(s["lo"]), 6),
                        round(float(s["hi"]), 6), 1500, "conditional within-year percentile resampling"])
    print("wrote out/fig_drift_curves.png + out/drift_resampling.csv")


# ----------------------------------------------------------------------------
# pipeline - Figure 1: dual-track schematic (left to right). Anchor along the top
#   (query encoder -> g), abstract A_i along the bottom (article encoder -> e_i);
#   the tracks merge into cosine -> annual mean -> Kendall tau -> three directions.
# ----------------------------------------------------------------------------
def run_pipeline():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(16.6, 6.6))
    ax.set_xlim(0, 16.6); ax.set_ylim(0, 6); ax.axis("off")
    yT, yB, yM = 4.2, 1.8, 3.0
    c1, c2, c3 = 1.5, 3.85, 6.2                      # tracks (compact)
    c4, c5, c6 = 8.55, 10.75, 12.95                  # merge chain
    cx = 15.25                                       # outcome chips (roomy gap)
    EDGE, ENC, TXT = "#33475b", "#cfe0ef", "#16222e"
    hw_t, hw_m, hw_c = 0.975, 0.95, 0.925

    def box(x, y, text, w=1.95, h=1.0, fc="white", ec=EDGE, tc=TXT, fs=11.5, bold=False, stack=False, lw=1.7):
        if stack:                                    # "many" - faint offset copies behind
            for k in (2, 1):
                o = 0.13 * k
                ax.add_patch(FancyBboxPatch((x - w / 2 + o, y - h / 2 + o), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.12", fc="#fbfcfd", ec="#a7b4c1", lw=1.0, zorder=1))
        ax.add_patch(FancyBboxPatch((x - w / 2 + 0.05, y - h / 2 - 0.07), w, h,   # soft shadow
                     boxstyle="round,pad=0.02,rounding_size=0.12", fc="#d7dde3", ec="none", zorder=1.4))
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                     fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", zorder=3)

    def arr(x1, y1, x2, y2, color="#5a6b7b", lw=2.3):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=1, shrinkB=1), zorder=1)

    for yc in (yT, yB):                              # faint bands group the two tracks
        ax.add_patch(FancyBboxPatch((0.4, yc - 0.82), 6.85, 1.64, boxstyle="round,pad=0,rounding_size=0.2",
                     fc="#f3f6f9", ec="none", zorder=0))
    ax.text(0.55, yT + 0.72, "anchor (query) track - one recommendation", fontsize=9.5, style="italic", color="#9aa7b3", va="center")
    ax.text(0.55, yB - 0.72, "literature (article) track - every record i", fontsize=9.5, style="italic", color="#9aa7b3", va="center")

    # top track (single anchor) / bottom track (MANY abstracts -> stacked)
    box(c1, yT, "Recommendation\nanchor (text)")
    box(c2, yT, "MedCPT\nquery encoder", fc=ENC, lw=2.0)
    box(c3, yT, "anchor vector\n$g$  (768-d)")
    box(c1, yB, "PubMed record\n$A_i$ in year $y$", stack=True)
    box(c2, yB, "MedCPT\narticle encoder", fc=ENC, lw=2.0)
    box(c3, yB, "article vectors\n$e_i$  (768-d)", stack=True)
    # merge chain
    box(c4, yM, "cosine\n$\\cos(g,\\,e_i)$", w=1.9, h=0.95, stack=True)
    box(c5, yM, "annual mean\ncosine, per year", w=1.9, h=0.95)
    box(c6, yM, "Kendall $\\tau$\ntrend vs year", w=1.9, h=0.95)
    # outcome chips (palette)
    box(cx, 4.30, "divergence\n$\\tau < 0$", w=1.85, h=0.74, fc=DIR_BLUE, ec=DIR_BLUE, tc="white", fs=10.5, bold=True)
    box(cx, 3.00, "near-zero trend\n$\\tau \\approx 0$", w=1.85, h=0.74, fc=DIR_GREY, ec=DIR_GREY, tc="white", fs=10.5, bold=True)
    box(cx, 1.70, "convergence\n$\\tau > 0$", w=1.85, h=0.74, fc=DIR_ORANGE, ec=DIR_ORANGE, tc="white", fs=10.5, bold=True)

    arr(c1 + hw_t, yT, c2 - hw_t, yT); arr(c2 + hw_t, yT, c3 - hw_t, yT)
    arr(c1 + hw_t, yB, c2 - hw_t, yB); arr(c2 + hw_t, yB, c3 - hw_t, yB)
    arr(c3 + hw_t, yT, c4 - hw_c, yM + 0.22)         # tracks merge into the cosine
    arr(c3 + hw_t, yB, c4 - hw_c, yM - 0.22)
    arr(c4 + hw_m, yM, c5 - hw_m, yM); arr(c5 + hw_m, yM, c6 - hw_m, yM)
    ax.text((c4 + c5) / 2, yM - 0.78, "averaged\nwithin each year", fontsize=8.3, style="italic",
            color="#8a97a3", ha="center", va="center")
    for yo in (4.30, 3.00, 1.70):                    # fan to the three outcomes
        arr(c6 + hw_m, yM, cx - 0.925, yo, color="#8a97a3", lw=2.0)

    fig.savefig(f"{OUT}/fig_pipeline.png", dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.rcParams["svg.fonttype"] = "none"   # keep plain labels as editable <text> in the SVG
    plt.rcParams["svg.hashsalt"] = "tide"   # deterministic element ids so the SVG is byte-reproducible
    fig.savefig(f"{OUT}/fig_pipeline.svg", bbox_inches="tight", pad_inches=0.15, metadata={"Date": None})
    print("wrote out/fig_pipeline.png + out/fig_pipeline.svg (dual-track schematic v2)")


SUBS = {"metrics": run_metrics, "figure": run_figure, "pipeline": run_pipeline}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (SUBS if which == "all" else [which]):
        print(f"\n########## drift.py :: {name} ##########")
        SUBS[name]()
