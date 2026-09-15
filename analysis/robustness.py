"""Frozen-vector sensitivity: null|bins|voladj|ols|loo|all (default all)."""
import os, sys, csv
import numpy as np
from scipy.stats import kendalltau, t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from drift import annual_mean_cos
from source import CASES as PRIMARY_CASES

np.random.seed(0)
CASES = [(c["case"], c["trial_id"], c["expected_direction"]) for c in PRIMARY_CASES]
Y0, Y1, MINPUB = 2000, 2024, 5


def abstract_cos(d):
    """Per-abstract cosine to the anchor, kept to years with >= MINPUB abstracts (2000-2024)."""
    yr, cos = annual_mean_cos({"d": d})
    keep = {y for y in np.unique(yr) if (yr == y).sum() >= MINPUB}
    mm = np.array([y in keep for y in yr])
    return yr[mm], cos[mm]


def series(d, floor):
    """Annual mean cosine over years meeting the abstract-count floor, plus per-year counts."""
    yr, cos = annual_mean_cos({"d": d})
    ys = [y for y in np.unique(yr) if (yr == y).sum() >= floor]
    means = np.array([cos[yr == y].mean() for y in ys])
    counts = {int(y): int((yr == y).sum()) for y in np.unique(yr)}
    return np.array(ys), means, counts


def phase_surrogate(x, rng):
    """autocorrelation-preserving surrogate: randomize Fourier phases, preserve power spectrum."""
    x = np.asarray(x, float); n = len(x); m = x.mean()
    X = np.fft.rfft(x - m)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(X))); ph[0] = 1.0
    if n % 2 == 0:
        ph[-1] = 1.0
    return np.fft.irfft(np.abs(X) * ph, n) + m


def run_null(B=5000):
    rng = np.random.default_rng(0)
    rows = []
    print(f"{'case':5} {'n_yrs':>5} {'tau':>8} {'phase_p':>9} {'expected':>12}")
    for name, d, exp in CASES:
        ys, means, _ = series(d, 5)
        tau = kendalltau(ys, means)[0]
        cnt = 0
        for _ in range(B):
            s = phase_surrogate(means, rng)
            if abs(kendalltau(ys, s)[0]) >= abs(tau) - 1e-12:
                cnt += 1
        p = (cnt + 1) / (B + 1)
        print(f"{name:5} {len(ys):>5} {tau:>+8.3f} {p:>9.4f} {exp:>12}")
        rows.append((name, len(ys), round(float(tau), 3), round(p, 4), p, cnt, B, exp))
    with open(f"{OUT}/robustness_null.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "n_yrs", "tau", "phase_randomized_p", "phase_exact_p", "exceedances", "replicates", "expected"])
        w.writerows(rows)
    print("wrote out/robustness_null.csv")


def run_bins():
    rows = []
    print(f"\n{'case':5} {'floor':>5} {'n_yrs':>5} {'tau':>8} {'dir':>5}")
    for name, d, exp in CASES:
        _, _, counts = series(d, 5)
        rng_yrs = sorted(counts)
        print(f"  {name} per-year counts {rng_yrs[0]}-{rng_yrs[-1]}: "
              f"min={min(counts.values())} median={int(np.median(list(counts.values())))} max={max(counts.values())}")
        for floor in (5, 10, 20):
            ys, means, _ = series(d, floor)
            if len(ys) < 4:
                print(f"{name:5} {floor:>5} {len(ys):>5}  (too few yrs)"); rows.append((name, floor, len(ys), "", "")); continue
            tau = kendalltau(ys, means)[0]
            dr = "div" if tau < -0.1 else ("conv" if tau > 0.1 else "flat")
            print(f"{name:5} {floor:>5} {len(ys):>5} {tau:>+8.3f} {dr:>5}")
            rows.append((name, floor, len(ys), round(float(tau), 3), dr))
    with open(f"{OUT}/robustness_bincount.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "min_count_floor", "n_yrs", "tau", "direction"]); w.writerows(rows)
    print("wrote out/robustness_bincount.csv")


def partial_kendall(x, y, z):
    """Partial Kendall's tau of x,y controlling for z (Kendall 1942)."""
    txy = kendalltau(x, y)[0]; txz = kendalltau(x, z)[0]; tyz = kendalltau(y, z)[0]
    return (txy - txz * tyz) / np.sqrt((1 - txz ** 2) * (1 - tyz ** 2)), txy, txz, tyz


def ols_year_adj(yr, mc, vol):
    """OLS: mean-cosine ~ 1 + z(year) + z(log volume). Return year and volume betas + two-sided p."""
    z = lambda a: (a - a.mean()) / a.std()
    X = np.column_stack([np.ones(len(yr)), z(yr.astype(float)), z(np.log(vol))])
    beta, *_ = np.linalg.lstsq(X, mc, rcond=None)
    dof = len(yr) - X.shape[1]
    resid = mc - np.einsum("ij,j->i", X, beta)
    rss = np.einsum("i,i->", resid, resid)
    xtx = np.einsum("ij,ik->jk", X, X)
    se = np.sqrt(np.diag((rss / dof) * np.linalg.inv(xtx)))
    p = lambda i: float(2 * (1 - tdist.cdf(abs(beta[i] / se[i]), dof)))
    return float(beta[1]), p(1), float(beta[2]), p(2)


def run_voladj():
    rows = []
    print(f"\n{'case':5} {'nyr':>3} {'t(yr,cos)':>9} {'t(yr,vol)':>9} {'t(cos,vol)':>10} "
          f"{'partial':>8} {'b_year':>8} {'p_yr':>6} {'b_vol':>8} {'p_vol':>6} {'dir':>5}")
    for name, d, exp in CASES:
        ys, means, counts = series(d, 5)
        vol = np.array([counts[int(y)] for y in ys], float)
        pt, txy, txz, tyz = partial_kendall(ys.astype(float), means, vol)
        by, py, bv, pv = ols_year_adj(ys, means, vol)
        dr = "div" if pt < -0.05 else ("conv" if pt > 0.05 else "flat")
        print(f"{name:5} {len(ys):>3} {txy:>+9.3f} {txz:>+9.3f} {tyz:>+10.3f} "
              f"{pt:>+8.3f} {by:>+8.4f} {py:>6.3f} {bv:>+8.4f} {pv:>6.3f} {dr:>5}")
        # Retain calculation precision: coarse rounding creates artificial jumps near .xxx5.
        rows.append((name, len(ys), round(txy, 3), round(txz, 3), round(tyz, 3), round(pt, 3),
                     round(by, 8), round(py, 6), round(bv, 8), round(pv, 6), dr, exp))
    with open(f"{OUT}/robustness_voladj.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "n_yrs", "tau_year_cos", "tau_year_vol", "tau_cos_vol",
                    "partial_tau_year_cos_given_vol", "ols_beta_year", "ols_p_year",
                    "ols_beta_vol", "ols_p_vol", "partial_direction", "expected"])
        w.writerows(rows)
    print("wrote out/robustness_voladj.csv")
    print("NOTE: year and volume are strongly collinear; adjusted estimates are imprecise. "
          "These descriptive checks do not establish absence of confounding.")


def cluster_robust_ols(yr, cos):
    """Abstract-level cosine ~ 1 + year, standard errors clustered by year (CR1).
    Returns (slope_per_year, cluster_robust_two_sided_p)."""
    x = yr.astype(float) - np.mean(yr)
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, cos, rcond=None)
    resid = cos - np.einsum("ij,j->i", X, beta)
    XtX_inv = np.linalg.inv(np.einsum("ij,ik->jk", X, X))
    groups = np.unique(yr)
    meat = np.zeros((2, 2))
    for g in groups:
        Xg = X[yr == g]; ug = resid[yr == g]
        s = np.einsum("ij,i->j", Xg, ug)
        meat += np.outer(s, s)
    G, N, K = len(groups), len(cos), 2
    correction = (G / (G - 1)) * ((N - 1) / (N - K))
    V = correction * np.einsum("ij,jk,kl->il", XtX_inv, meat, XtX_inv)
    se = np.sqrt(V[1, 1])
    p = float(2 * (1 - tdist.cdf(abs(beta[1] / se), G - 1)))
    return float(beta[1]), p


def run_ols():
    rows = []
    print(f"{'case':5} {'n_abs':>6} {'n_yrs':>5} {'slope(1e-4/yr)':>15} {'cluster_p':>11} {'expected':>12}")
    for name, d, exp in CASES:
        yr, cos = abstract_cos(d)
        slope, p = cluster_robust_ols(yr, cos)
        print(f"{name:5} {len(cos):>6} {len(np.unique(yr)):>5} {slope * 1e4:>15.2f} {p:>11.2e} {exp:>12}")
        rows.append((name, len(cos), len(np.unique(yr)), round(slope, 8), round(slope * 1e4, 2), f"{p:.2e}", exp))
    with open(f"{OUT}/robustness_ols.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "n_abstracts", "n_yrs", "ols_slope_cos_per_yr", "ols_slope_1e4_per_yr",
                    "cluster_robust_p", "expected"])
        w.writerows(rows)
    print("wrote out/robustness_ols.csv")


def run_loo():
    rows = []
    print(f"\n{'case':5} {'n_yrs':>5} {'full_tau':>9} {'loo_min':>9} {'loo_max':>9} {'expected':>12}")
    for name, d, exp in CASES:
        ys, means, _ = series(d, 5)
        full = kendalltau(ys, means)[0]
        loo = [kendalltau(np.delete(ys, i), np.delete(means, i))[0] for i in range(len(ys))]
        lo, hi = float(min(loo)), float(max(loo))
        print(f"{name:5} {len(ys):>5} {full:>+9.3f} {lo:>+9.3f} {hi:>+9.3f} {exp:>12}")
        rows.append((name, len(ys), round(float(full), 3), round(lo, 3), round(hi, 3), exp))
    with open(f"{OUT}/robustness_loo.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "n_yrs", "full_tau", "loo_tau_min", "loo_tau_max", "expected"]); w.writerows(rows)
    print("wrote out/robustness_loo.csv")


SUBS = {"null": run_null, "bins": run_bins, "voladj": run_voladj, "ols": run_ols, "loo": run_loo}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (SUBS if which == "all" else [which]):
        SUBS[name]()
