"""
data.py — Loading, cleaning and batching of NYC Uber pickup data.

Dataset: "Uber Pickups in New York City" (Kaggle, fivethirtyeight).
Expected raw files in data/raw/:
    uber-raw-data-apr14.csv ... uber-raw-data-sep14.csv
Columns: Date/Time, Lat, Lon, Base

Design note
-----------
Only Lat/Lon are fed to the model. The timestamp is NEVER a model feature:
it is used to (a) split the data into monthly production batches and
(b) build the hourly demand profile of each discovered zone.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

# Training batch = the model that first goes to production.
TRAIN_MONTHS = ["apr14", "may14"]
# Production batches = "the future" that arrives one month per pipeline run.
STREAM_MONTHS = ["jun14", "jul14", "aug14", "sep14"]

# NYC bounding box. Drops (0, 0) rows and GPS garbage.
LAT_MIN, LAT_MAX = 40.50, 41.00
LON_MIN, LON_MAX = -74.30, -73.70

FEATURES = ["Lat", "Lon"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = df.dropna(subset=["Lat", "Lon"])
    mask = (
        df["Lat"].between(LAT_MIN, LAT_MAX)
        & df["Lon"].between(LON_MIN, LON_MAX)
    )
    df = df.loc[mask].copy()
    if "Date/Time" in df.columns:
        df["ts"] = pd.to_datetime(df["Date/Time"], errors="coerce")
        df = df.dropna(subset=["ts"])
        df["hour"] = df["ts"].dt.hour
        df["dow"] = df["ts"].dt.dayofweek  # 0 = Monday
    return df.reset_index(drop=True)


def load_months(months, sample: int | None = 400_000, seed: int = 42) -> pd.DataFrame:
    """Load one or more monthly CSVs, clean them and optionally subsample."""
    frames = []
    for m in months:
        path = RAW_DIR / f"uber-raw-data-{m}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download the Kaggle dataset "
                f"'Uber Pickups in New York City' into data/raw/."
            )
        frames.append(pd.read_csv(path))
    df = _clean(pd.concat(frames, ignore_index=True))
    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=seed).reset_index(drop=True)
    return df


def make_synthetic_batch(n: int = 20_000, n_zones: int = 5,
                         seed: int = 0, shift: float = 0.0,
                         centre_seed: int = 0) -> pd.DataFrame:
    """
    Synthetic pickups, used by the CI test-suite so the pipeline can run
    without the (2 GB) real dataset.

    `centre_seed` fixes the geography: two batches with the same centre_seed
    describe the same city, so a fresh batch must NOT trigger drift.
    `seed` only resamples points. `shift` displaces the centres, which is how
    spatial drift is injected on demand.
    """
    rng = np.random.default_rng(seed)
    centres = np.random.default_rng(centre_seed).uniform(
        [LAT_MIN + 0.05, LON_MIN + 0.05],
        [LAT_MAX - 0.05, LON_MAX - 0.05],
        size=(n_zones, 2),
    ) + shift
    idx = rng.integers(0, n_zones, size=n)
    pts = centres[idx] + rng.normal(0, 0.01, size=(n, 2))
    ts = pd.to_datetime("2014-04-01") + pd.to_timedelta(
        rng.integers(0, 60 * 24 * 30, size=n), unit="m"
    )
    return pd.DataFrame({
        "Lat": pts[:, 0], "Lon": pts[:, 1], "ts": ts,
        "hour": ts.hour, "dow": ts.dayofweek,
    })


def demand_profile(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Pickup counts per (zone, day-of-week, hour).
    This table is what the app actually queries; it must be regenerated on
    every retrain, otherwise the UI shows zones the model no longer has.
    """
    out = df.copy()
    out["zone"] = labels
    prof = (out.groupby(["zone", "dow", "hour"])
               .size().reset_index(name="pickups"))
    return prof


def save_processed(prof: pd.DataFrame, centroids: np.ndarray) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    prof.to_parquet(PROC_DIR / "demand_profile.parquet", index=False)
    pd.DataFrame(centroids, columns=["Lat", "Lon"]).to_csv(
        PROC_DIR / "centroids.csv", index=False
    )
