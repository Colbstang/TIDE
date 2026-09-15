#!/usr/bin/env python3
"""One-command reproduction and verification for the TIDE paper."""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
from packaging.version import Version


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "out"
SOURCE_FILES = (
    "analysis/boundary.py", "analysis/common.py", "analysis/drift.py", "analysis/comparisons.py",
    "analysis/interpretability.py", "analysis/paper.py", "analysis/pubmed.py",
    "analysis/query_sensitivity.py", "analysis/robustness.py", "analysis/sensitivity.py", "analysis/source.py",
    "tests/test_comparisons.py", "tests/test_paper.py", "tests/test_reproduction.py", "tests/test_sensitivity.py",
    "tests/test_public.py",
)

# Never package third-party article text, even when restored locally.
PRIVATE_TEXT = frozenset({
    "data/raw/trial_aaos_distal_radius_elderly_nonop_pubmed_2000_2024.json",
    "data/raw/trial_acute_compartment_fasciotomy_pubmed_2000_2024.json",
    "data/raw/trial_pji_two_stage_exchange_pubmed_2000_2024.json",
    "data/sensitivity/query_records.json.gz",
})
PUBLIC_COMMANDS = [
    ("analysis/drift.py", "metrics"), ("analysis/drift.py", "figure"),
    ("analysis/drift.py", "pipeline"), ("analysis/robustness.py", "all"),
]
PUBLIC_NUMERIC = [
    "case_study_metrics.csv", "drift_resampling.csv", "robustness_bincount.csv",
    "robustness_loo.csv", "robustness_null.csv", "robustness_ols.csv", "robustness_voladj.csv",
]

CORE_COMMANDS = [
    ("analysis/source.py", "audit"),
    ("analysis/sensitivity.py", "weighting"),
    ("analysis/query_sensitivity.py", "audit"),
    ("analysis/drift.py", "metrics"),
    ("analysis/drift.py", "figure"),
    ("analysis/drift.py", "pipeline"),
    ("analysis/robustness.py", "all"),
    ("analysis/comparisons.py",),
    ("analysis/paper.py", "core"),
]
MODEL_COMMANDS = [
    ("analysis/interpretability.py", "all"),
    ("analysis/boundary.py", "attention"),
    ("analysis/paper.py", "probes"),
]

CORE_NUMERIC = [
    "records.csv", "source_counts.csv", "annual_series.csv", "matched_baselines.csv",
    "evidence_weighting.csv", "evidence_weighting_annual.csv", "evidence_weights.csv",
    "query_sensitivity.csv", "query_sensitivity_annual.csv",
    "case_study_metrics.csv",
    "drift_resampling.csv",
    "robustness_bincount.csv",
    "robustness_loo.csv",
    "robustness_null.csv",
    "robustness_ols.csv",
    "robustness_voladj.csv",
    "paper_core.csv",
    "tfidf_variants.csv", "tfidf_annual.csv", "cross_anchors.csv", "cross_anchor_annual.csv",
]
MODEL_NUMERIC = [
    "minimal_pairs.csv", "geometry_summary.csv", "matched_cls.csv", "paper_probes.csv",
    "attention_reshaping.csv",
    "word_traj_PJI.csv",
    "pji_year_centroids.npy",
]
CORE_FIGURES = ["fig_drift_curves.png", "fig_pipeline.png", "fig_pipeline.svg"]
MODEL_FIGURES = [
    "fig_geometry_pji.png",
    "fig_word_trajectories_PJI.png",
]

EXACT_CSV_COLUMNS = {
    "case", "pmid", "year", "word", "pair", "direction", "expected", "status",
    "text_sha256", "raw_row", "vector_row", "analyzed", "n", "n_yrs", "n_years",
    "rows", "replicates", "exceedances", "min_count_floor", "seed",
}
EXACT_CSV_SUFFIXES = (
    "_id", "_sha256", "_row", "_records", "_count", "_counts",
    "_abstracts", "_replicates", "_exceedances", "_draws",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_environment(full):
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Use Python 3.13 for the frozen reproduction environment")
    packages = {}
    for name in ["requirements-core.txt"] + (["requirements.txt"] if full else []):
        for line in (ROOT / name).read_text().splitlines():
            if "==" not in line or line.startswith("#"):
                continue
            package_name, expected = line.split("==")
            try:
                actual = version(package_name)
            except PackageNotFoundError:
                raise SystemExit(f"Missing {package_name}; install {name}") from None
            if Version(actual).public != expected:
                raise SystemExit(f"{package_name}: installed {actual}, required {expected}; install {name}")
            packages[package_name] = actual
    return packages


def verify_inputs(with_text=False) -> None:
    provenance = json.loads((ROOT / "provenance.json").read_text())
    failures = []
    artifacts = dict(provenance["artifacts"])
    artifacts.update(provenance.get("reference_artifacts", {}))
    if not with_text:
        artifacts = {name: value for name, value in artifacts.items() if name not in PRIVATE_TEXT}
    for relative, expected in artifacts.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
        elif sha256(path) != expected:
            failures.append(f"checksum mismatch {relative}")
    if failures:
        raise SystemExit("Frozen-input verification failed:\n  " + "\n  ".join(failures)
                         + "\nText-dependent checks require the separate frozen text archive; see README.md.")
    print(f"verified {len(artifacts)} input/reference checksums", flush=True)


def run(commands, output: Path, offline: bool, embeddings=None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "TIDE_OUT": str(output),
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "TIDE_EMB": str(embeddings or ROOT / "data/embeddings"),
        "TOKENIZERS_PARALLELISM": "false",
    })
    if offline:
        env["TIDE_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    with tempfile.TemporaryDirectory(prefix="tide-matplotlib-") as mpl_config:
        env["MPLCONFIGDIR"] = mpl_config
        for command in commands:
            label = "+ python " + " ".join(command)
            print(label, flush=True)
            with (output / "run.log").open("a") as log:
                log.write(label + "\n")
                with subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    if process.wait():
                        raise RuntimeError(f"Failed: {label}; see {output / 'run.log'}")


def equivalent(generated: Path, reference: Path, atol=0.0) -> bool:
    """Exact by default; declared absolute tolerance for floating-point rebuilds.

    Schema, row ordering, identifiers, and text must always match. Nonfinite
    numbers fail. Never accept a missing field or silently drop a comparison.
    """
    identical = generated.read_bytes() == reference.read_bytes()
    if generated.suffix == ".npy":
        a, b = np.load(generated, allow_pickle=False), np.load(reference, allow_pickle=False)
        return (a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                and (identical or (atol > 0 and np.allclose(a, b, atol=atol, rtol=0))))
    if generated.suffix == ".csv":
        with generated.open(newline="") as ga, reference.open(newline="") as rb:
            a, b = list(csv.reader(ga)), list(csv.reader(rb))
        if (not a or len(a) != len(b) or a[0] != b[0] or not a[0]
                or any(not column for column in a[0]) or len(set(a[0])) != len(a[0])):
            return False
        # Validate even identical files: equality alone cannot certify usable data.
        for row in a[1:] + b[1:]:
            if len(row) != len(a[0]):
                return False
            for value in row:
                try:
                    if not np.isfinite(float(value)):
                        return False
                except ValueError:
                    pass  # Text and deliberately blank cells are allowed.
        if identical:
            return True
        if atol == 0:
            return False
        for xrow, yrow in zip(a[1:], b[1:]):
            for column, x, y in zip(a[0], xrow, yrow):
                if x == y:
                    continue
                # Identifiers and declared integer/count fields must match exactly.
                if (column in EXACT_CSV_COLUMNS or column.endswith(EXACT_CSV_SUFFIXES)
                        or re.fullmatch(r"[+-]?\d+", y)):
                    return False
                try:
                    xv, yv = float(x), float(y)
                except ValueError:
                    return False
                limit = atol
                if column.endswith("_p") or column.startswith("ols_p_"):
                    # Tiny p-values must not pass solely because they are <1e-5.
                    limit = min(atol, max(1e-12, abs(yv) * 0.005))
                # np.isclose gives the declared boundary only machine-scale
                # slack, avoiding false failures when a decimal difference of
                # exactly ``limit`` is represented a few ulps above it.
                if (not np.isfinite([xv, yv]).all()
                        or not np.isclose(xv, yv, atol=limit, rtol=1e-12)):
                    return False
        return True
    return identical


def verify_outputs(output: Path, full: bool, atol=0.0, public_only=False) -> None:
    numeric = PUBLIC_NUMERIC if public_only else CORE_NUMERIC + (MODEL_NUMERIC if full else [])
    figures = CORE_FIGURES + (MODEL_FIGURES if full else [])
    failures = []
    for name in numeric:
        generated, reference = output / name, REFERENCE / name
        if not generated.is_file():
            failures.append(f"missing generated output {name}")
        elif not reference.is_file():
            failures.append(f"missing frozen reference {name}")
        elif not equivalent(generated, reference, atol):
            failures.append(f"does not match frozen reference: {name}")
    for name in figures:
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty figure {name}")
    if failures:
        raise SystemExit("Reproduction failed:\n  " + "\n  ".join(failures))
    scope = "public frozen-vector subset" if public_only else "source-to-output" if full else "text-assisted core"
    comparison = f"absolute tolerance {atol:g}" if atol else "byte-for-byte"
    print(f"PASS: {scope} reproduction matches reference numerical outputs ({comparison})", flush=True)
    if not full:
        print("run `python reproduce.py --full` to rebuild all primary vectors and verify model-dependent outputs")


def package(path: Path) -> None:
    """Public allowlist: no article text, Git history, caches or logs."""
    provenance = json.loads((ROOT / "provenance.json").read_text())
    files = {"README.md", "LICENSE", "CITATION.cff", "anchors.json", "provenance.json",
             "reproduce.py", "requirements.txt", "requirements-core.txt", ".gitignore", ".gitattributes", ".github/workflows/reproduce.yml"}
    files.update(provenance["artifacts"])
    files.update(provenance["reference_artifacts"])
    files.update(SOURCE_FILES)
    files.difference_update(PRIVATE_TEXT)
    # Fixed ZIP metadata makes the package deterministic; exclusive create protects existing releases.
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(files):
            info = zipfile.ZipInfo("TIDE/" + relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT / relative).read_bytes())
    print(f"Package: {path}\nSHA256: {sha256(path)}")


def check(output, full, offline, with_text=False):
    tests = ("-m", "unittest", "discover", "-s", "tests", "-v")
    if not with_text:
        tests += ("-p", "test_public.py")
    run([tests], output, offline)
    run(CORE_COMMANDS if with_text else PUBLIC_COMMANDS, output, offline)
    verify_outputs(output, False, public_only=not with_text)
    if full:
        run([("analysis/source.py", "rebuild")], output, offline)
        if not equivalent(output / "matched_cls.csv", REFERENCE / "matched_cls.csv", atol=1e-5):
            raise ValueError("Matched-record CLS audit differs from reference")
        run([("analysis/query_sensitivity.py", "rebuild")], output, offline)
        for name in ("query_sensitivity.csv", "query_sensitivity_annual.csv"):
            if not equivalent(output / name, REFERENCE / name, atol=1e-5):
                raise ValueError(f"Rebuilt query sensitivity differs: {name}")
        derived = output / "from-rebuilt"
        derived.mkdir()
        shutil.copyfile(output / "matched_cls.csv", derived / "matched_cls.csv")
        run(CORE_COMMANDS + MODEL_COMMANDS, derived, offline, embeddings=output / "rebuilt")
        verify_outputs(derived, True, atol=1e-5)
    (output / "verification.json").write_text(json.dumps(dict(
        status="PASS", scope=("archived-source-to-output" if full else
                             "text-assisted-core" if with_text else "public-frozen-vector-subset"),
        python=sys.version, full=full, offline=offline, text_archive_used=with_text,
        verified_numeric_outputs=(CORE_NUMERIC + (MODEL_NUMERIC if full else [])) if with_text else PUBLIC_NUMERIC,
        packages=verify_environment(full),
        note="Public mode verifies primary trends and vector-only robustness. Other analyses require the separate text archive. No mode certifies the original search event or clinical validity."), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-text", action="store_true", help="verify all core analyses after restoring the separate frozen text archive")
    parser.add_argument("--full", action="store_true", help="rebuild every primary vector from archived text and rerun all implemented analyses")
    parser.add_argument("--offline", action="store_true", help="forbid model downloads; require a populated cache")
    parser.add_argument("--output", type=Path, help="keep verification logs and regenerated artifacts in a NEW directory")
    parser.add_argument("--package", type=Path, help="after verification, create a public code/vector/result ZIP without article text")
    args = parser.parse_args()

    if args.package and args.package.exists():
        parser.error("package destination already exists")
    verify_environment(args.full)
    with_text = args.with_text or args.full
    verify_inputs(with_text)
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        check(output, args.full, args.offline, with_text)
    else:
        with tempfile.TemporaryDirectory(prefix="tide-reproduce-") as directory:
            check(Path(directory), args.full, args.offline, with_text)
    if args.package:
        package(args.package.resolve())


if __name__ == "__main__":
    main()
