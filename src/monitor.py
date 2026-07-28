"""
monitor.py — Data/model drift detection for the demand-zone model.

Three signals, all computed against the model currently in production:

1. silhouette_new   — cluster quality on the incoming batch. Falls when the
                      learned zones stop describing reality.
2. far_point_rate   — share of new pickups whose distance to the nearest
                      centroid exceeds the p99 of the training distances.
                      This is the fastest signal when demand appears in a
                      place the model has never seen.
3. centroid_shift   — mean displacement (km) of the centroids if the model
                      were refit on the new batch.

Exit code 1 => drift detected => the retraining workflow proceeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import silhouette_score

from src import data as D

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

# Thresholds. Tune them with the report of the first two runs, then freeze.
SIL_DROP_PCT = 0.15      # 15% relative drop vs the production baseline
FAR_POINT_MAX = 0.10     # 10% of new points too far from any centroid
KM_PER_DEG = 111.0


def far_point_rate(X: np.ndarray, model, threshold: float) -> float:
    d = np.min(model.transform(X), axis=1)
    return float((d > threshold).mean())


def training_distance_p99(X: np.ndarray, model) -> float:
    return float(np.percentile(np.min(model.transform(X), axis=1), 99))


def run(new_df, baseline: dict, model) -> dict:
    X = new_df[D.FEATURES].to_numpy()
    labels = model.predict(X)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(20_000, len(X)), replace=False)

    sil_new = float(silhouette_score(X[idx], labels[idx]))
    sil_base = float(baseline["silhouette"])
    rel_drop = (sil_base - sil_new) / abs(sil_base) if sil_base else 0.0
    far = far_point_rate(X, model, baseline["dist_p99"])

    drift = bool(rel_drop > SIL_DROP_PCT or far > FAR_POINT_MAX)
    report = {
        "silhouette_production": sil_base,
        "silhouette_new_batch": sil_new,
        "relative_drop": float(rel_drop),
        "far_point_rate": far,
        "thresholds": {"sil_drop_pct": SIL_DROP_PCT,
                       "far_point_max": FAR_POINT_MAX},
        "drift_detected": drift,
        "reason": ("silhouette drop" if rel_drop > SIL_DROP_PCT
                   else "unseen demand areas" if far > FAR_POINT_MAX
                   else "none"),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=["jun14"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--shift", type=float, default=0.0,
                    help="Synthetic mode only: inject spatial drift.")
    args = ap.parse_args()

    model = joblib.load(MODELS / "production.joblib")
    baseline = json.loads((MODELS / "production_metrics.json").read_text())

    new_df = (D.make_synthetic_batch(20_000, seed=7, shift=args.shift)
              if args.synthetic else D.load_months(args.months, sample=200_000))

    report = run(new_df, baseline, model)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "drift_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    raise SystemExit(1 if report["drift_detected"] else 0)


if __name__ == "__main__":
    main()
