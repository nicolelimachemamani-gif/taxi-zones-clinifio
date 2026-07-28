"""
Smoke tests for the CI workflow. They run on synthetic data so the pipeline
is testable without the 2 GB Kaggle dataset (GitHub Actions has no Kaggle
credentials, and committing the data to the repo would be wrong anyway).
"""
import joblib
import numpy as np
import pandas as pd
import pytest

from src import data as D
from src import monitor
from src.train import evaluate, fit_final


@pytest.fixture(scope="module")
def batch():
    return D.make_synthetic_batch(5_000, n_zones=5, seed=1)


@pytest.fixture(scope="module")
def model(batch):
    return fit_final(batch[D.FEATURES].to_numpy(), k=5)


def test_cleaning_drops_out_of_bounds():
    bad = pd.DataFrame({"Date/Time": ["4/1/2014 0:11", "4/1/2014 0:17"],
                        "Lat": [40.75, 0.0], "Lon": [-73.98, 0.0]})
    assert len(D._clean(bad)) == 1


def test_features_are_only_coordinates():
    assert D.FEATURES == ["Lat", "Lon"], "The timestamp must not be a feature."


def test_model_trains_and_predicts(batch, model):
    X = batch[D.FEATURES].to_numpy()
    labels = model.predict(X)
    assert len(labels) == len(X)
    assert set(np.unique(labels)).issubset(set(range(5)))


def test_silhouette_is_reasonable(batch, model):
    m = evaluate(batch[D.FEATURES].to_numpy(), model)
    assert m["silhouette"] > 0.3


def test_demand_profile_covers_all_zones(batch, model):
    labels = model.predict(batch[D.FEATURES].to_numpy())
    prof = D.demand_profile(batch, labels)
    assert prof["pickups"].sum() == len(batch)
    assert set(prof["zone"]) == set(np.unique(labels))


def test_model_roundtrip(tmp_path, batch, model):
    p = tmp_path / "m.joblib"
    joblib.dump(model, p)
    reloaded = joblib.load(p)
    X = batch[D.FEATURES].to_numpy()[:100]
    assert (reloaded.predict(X) == model.predict(X)).all()


def test_drift_not_flagged_on_similar_batch(batch, model):
    X = batch[D.FEATURES].to_numpy()
    baseline = evaluate(X, model)
    baseline["dist_p99"] = monitor.training_distance_p99(X, model)
    same = D.make_synthetic_batch(5_000, n_zones=5, seed=1)
    assert monitor.run(same, baseline, model)["drift_detected"] is False


def test_drift_flagged_on_shifted_batch(batch, model):
    X = batch[D.FEATURES].to_numpy()
    baseline = evaluate(X, model)
    baseline["dist_p99"] = monitor.training_distance_p99(X, model)
    shifted = D.make_synthetic_batch(5_000, n_zones=5, seed=1, shift=0.08)
    assert monitor.run(shifted, baseline, model)["drift_detected"] is True
