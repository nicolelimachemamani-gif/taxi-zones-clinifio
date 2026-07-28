"""
train.py — Trains the demand-zone model (MiniBatchKMeans over Lat/Lon).

Usage
-----
    python -m src.train                      # initial training (apr+may)
    python -m src.train --months jun14       # retraining with a new batch
    python -m src.train --synthetic          # CI mode, no dataset needed

The script writes a *candidate* model to models/candidate.joblib and a JSON
metric card. Promotion to production is decided by promote.py, never here:
training and promotion are separate steps on purpose, so the pipeline can
refuse a worse model.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from src import data as D

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
K_GRID = list(range(15, 56, 5))
SIL_SAMPLE = 20_000  # silhouette on the full set is O(n^2): always subsample


def search_k(X: np.ndarray, k_grid=K_GRID, seed: int = 42):
    """Pick K by silhouette score. Returns (best_k, history)."""
    history = []
    for k in k_grid:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed,
                             n_init=5, batch_size=4096)
        labels = km.fit_predict(X)
        idx = np.random.default_rng(seed).choice(
            len(X), size=min(SIL_SAMPLE, len(X)), replace=False
        )
        sil = silhouette_score(X[idx], labels[idx])
        history.append({"k": int(k), "silhouette": float(sil),
                        "inertia": float(km.inertia_)})
        print(f"  K={k:>3}  silhouette={sil:.4f}  inertia={km.inertia_:.2f}")
    best = max(history, key=lambda h: h["silhouette"])
    return best["k"], history


def fit_final(X: np.ndarray, k: int, seed: int = 42) -> MiniBatchKMeans:
    return MiniBatchKMeans(n_clusters=k, random_state=seed,
                           n_init=10, batch_size=4096).fit(X)


def evaluate(X: np.ndarray, model) -> dict:
    labels = model.predict(X)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(SIL_SAMPLE, len(X)), replace=False)
    return {
        "k": int(model.n_clusters),
        "silhouette": float(silhouette_score(X[idx], labels[idx])),
        "calinski_harabasz": float(calinski_harabasz_score(X[idx], labels[idx])),
        "inertia": float(model.inertia_),
        "n_samples": int(len(X)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=D.TRAIN_MONTHS)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sample", type=int, default=400_000)
    ap.add_argument("--fixed-k", type=int, default=None,
                    help="Skip the K search (faster retrains).")
    args = ap.parse_args()

    df = (D.make_synthetic_batch(20_000) if args.synthetic
          else D.load_months(args.months, sample=args.sample))
    X = df[D.FEATURES].to_numpy()
    print(f"Training on {len(X):,} pickups from {args.months}")

    if args.fixed_k:
        k, history = args.fixed_k, []
    else:
        print("Searching K by silhouette...")
        k, history = search_k(X, k_grid=([3, 5, 8] if args.synthetic else K_GRID))
    print(f"Selected K = {k}")

    model = fit_final(X, k)
    metrics = evaluate(X, model)
    metrics.update({
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "months": args.months,
        "k_search": history,
    })

    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "candidate.joblib")
    (MODELS / "candidate_metrics.json").write_text(json.dumps(metrics, indent=2))

    labels = model.predict(X)
    D.save_processed(D.demand_profile(df, labels), model.cluster_centers_)
    print(json.dumps({k_: v for k_, v in metrics.items() if k_ != "k_search"},
                     indent=2))


if __name__ == "__main__":
    main()
