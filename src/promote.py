"""
promote.py — The quality gate. This file is what turns a training script
into a maintenance pipeline.

A candidate model replaces production ONLY if it is measurably better on the
most recent batch. Otherwise the current model stays and the workflow exits
without deploying anything.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from src import data as D
from src.train import evaluate

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ARCHIVE = MODELS / "archive"
MIN_GAIN = 0.005  # candidate must beat production by at least this much


def bootstrap_production() -> None:
    """First run: there is no production model yet, so the candidate becomes it."""
    shutil.copy(MODELS / "candidate.joblib", MODELS / "production.joblib")
    m = json.loads((MODELS / "candidate_metrics.json").read_text())
    m["dist_p99"] = m.get("dist_p99", 0.0)
    m["version"] = 1
    m["promoted_at"] = datetime.now(timezone.utc).isoformat()
    (MODELS / "production_metrics.json").write_text(json.dumps(m, indent=2))


def main() -> None:
    cand_path = MODELS / "candidate.joblib"
    prod_path = MODELS / "production.joblib"
    if not cand_path.exists():
        raise SystemExit("No candidate model to evaluate.")

    cand = joblib.load(cand_path)
    cand_metrics = json.loads((MODELS / "candidate_metrics.json").read_text())

    # Distance p99 of the candidate on its own training data: the reference
    # the drift monitor will use for the "far point" signal.
    df = D.load_months(cand_metrics["months"], sample=200_000) \
        if (ROOT / "data" / "raw").exists() and any(
            (ROOT / "data" / "raw").glob("*.csv")) else D.make_synthetic_batch(20_000)
    X = df[D.FEATURES].to_numpy()
    cand_metrics["dist_p99"] = float(
        np.percentile(np.min(cand.transform(X), axis=1), 99))

    if not prod_path.exists():
        joblib.dump(cand, prod_path)
        cand_metrics["version"] = 1
        cand_metrics["promoted_at"] = datetime.now(timezone.utc).isoformat()
        (MODELS / "production_metrics.json").write_text(
            json.dumps(cand_metrics, indent=2))
        print("PROMOTED: bootstrap, no previous production model.")
        return

    prod = joblib.load(prod_path)
    prod_on_new = evaluate(X, prod)["silhouette"]
    cand_on_new = evaluate(X, cand)["silhouette"]
    gain = cand_on_new - prod_on_new

    print(f"production silhouette = {prod_on_new:.4f}")
    print(f"candidate  silhouette = {cand_on_new:.4f}")
    print(f"gain = {gain:+.4f} (required > {MIN_GAIN})")

    if gain > MIN_GAIN:
        prev = json.loads((MODELS / "production_metrics.json").read_text())
        ARCHIVE.mkdir(exist_ok=True)
        v = int(prev.get("version", 1))
        shutil.copy(prod_path, ARCHIVE / f"production_v{v}.joblib")
        joblib.dump(cand, prod_path)
        cand_metrics["version"] = v + 1
        cand_metrics["promoted_at"] = datetime.now(timezone.utc).isoformat()
        cand_metrics["previous_silhouette"] = prod_on_new
        (MODELS / "production_metrics.json").write_text(
            json.dumps(cand_metrics, indent=2))
        print(f"PROMOTED: production is now v{v + 1}.")
    else:
        print("REJECTED: candidate is not better. Production model kept.")


if __name__ == "__main__":
    main()
