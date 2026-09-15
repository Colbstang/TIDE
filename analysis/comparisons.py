"""Post hoc matched TF-IDF variants and cross-anchor specificity checks."""
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from common import EMB
from source import CASES, analysis_mask, primary_analysis_records, save_csv, unit


def inputs(case):
    folder = Path(EMB) / case["trial_id"]
    vectors = unit(np.load(folder / "abstract_vectors.npy").astype(float))
    anchor = unit(np.load(folder / "guideline_vector.npy").astype(float).ravel())
    years = np.load(folder / "years.npy")
    keep = analysis_mask(years)
    return vectors[keep], years[keep], anchor


def components(vectors, years, anchor):
    """For unit vectors, mean cosine = centroid length * angular cosine."""
    ys = np.unique(years)
    centroids = np.array([vectors[years == y].mean(0) for y in ys])
    lengths = np.linalg.norm(centroids, axis=1)
    scores = np.einsum("ij,j->i", centroids, anchor)
    angles = scores / lengths
    direct = np.array([np.einsum("ij,j->i", vectors[years == y], anchor).mean()
                       for y in ys])
    np.testing.assert_allclose(lengths * angles, direct, atol=1e-12, rtol=0)
    return ys, centroids, scores, lengths, angles


def lexical_variants(texts, years, anchor):
    """Cross IDF fitting and scoring units; retain all four declared choices."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    years = np.asarray(years)
    if len(texts) != len(years):
        raise ValueError("Text/year rows must align")
    ys = np.unique(years)
    annual_text = [" ".join(t for t, y in zip(texts, years) if y == yr) for yr in ys]
    for fit_unit, fitting in (("year", annual_text), ("record", texts)):
        vectorizer = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)
        # Including the anchor preserves the historical baseline convention.
        vectorizer.fit([anchor] + list(fitting))
        reference = vectorizer.transform([anchor])
        annual = (vectorizer.transform(annual_text) @ reference.T).toarray().ravel()
        scores = (vectorizer.transform(texts) @ reference.T).toarray().ravel()
        means = np.array([scores[years == y].mean() for y in ys])
        for score_unit, values in (("year", annual), ("record_mean", means)):
            yield fit_unit, score_unit, ys, values


def comparisons():
    """September 2026 post hoc checks on the unchanged three primary inputs."""
    lexical, lexical_annual, cross, cross_annual = [], [], [], []
    cached = [inputs(case) for case in CASES]
    anchors = np.array([anchor for _, _, anchor in cached])
    for case, (vectors, years, own_anchor) in zip(CASES, cached):
        records = primary_analysis_records(case["case"])
        np.testing.assert_array_equal(years, [year for year, _ in records])
        ys = np.unique(years)
        for fit, score, _, values in lexical_variants(
                [text for _, text in records], years, case["anchor_text"]):
            tau = float(kendalltau(ys, values).statistic)
            if not np.isfinite(tau):
                raise ValueError(f"Undefined lexical trend: {case['case']}/{fit}/{score}")
            key = dict(case=case["case"], fit_unit=fit, score_unit=score)
            lexical.append(dict(**key, n=len(years), n_years=len(ys), tau=round(tau, 8)))
            lexical_annual.extend(dict(**key, year=int(y), mean_cosine=round(float(v), 8))
                                  for y, v in zip(ys, values))
        _, centroids, _, _, _ = components(vectors, years, own_anchor)
        for other, anchor in zip(CASES, anchors):
            values = np.einsum("ij,j->i", centroids, anchor)
            direct = np.array([np.einsum("ij,j->i", vectors[years == y], anchor).mean()
                               for y in ys])
            np.testing.assert_allclose(values, direct, atol=1e-12, rtol=0)
            tau = float(kendalltau(ys, values).statistic)
            if not np.isfinite(tau):
                raise ValueError(f"Undefined cross-anchor trend: {case['case']}/{other['case']}")
            key = dict(case=case["case"], anchor=other["case"])
            cross.append(dict(**key, n=len(years), n_years=len(ys), tau=round(tau, 8)))
            cross_annual.extend(dict(**key, year=int(y), mean_cosine=round(float(v), 8))
                                for y, v in zip(ys, values))
    for name, rows in (("tfidf_variants", lexical), ("tfidf_annual", lexical_annual),
                       ("cross_anchors", cross), ("cross_anchor_annual", cross_annual)):
        save_csv(name + ".csv", rows)
    print("Wrote all four TF-IDF variants and all nine case/anchor comparisons")


if __name__ == "__main__":
    comparisons()
