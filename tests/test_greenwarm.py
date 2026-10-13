import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from greenwarm.carbon import REGIONS, ci_series
from greenwarm.experiment import run_all
from greenwarm.predictor import MeanRatePredictor, GBMRatePredictor
from greenwarm.simulator import Policy, simulate, carbon_factor
from greenwarm.traffic import synthetic_dataset, load_azure_trace


def test_carbon_series_shape_and_wrap():
    for key in REGIONS:
        s = ci_series(key)
        assert s.shape == (1440,) and (s > 0).all()


def test_carbon_factor_direction():
    assert carbon_factor(1, 1, 400, 500) < 1       # clean hour: keep warm more easily
    assert carbon_factor(1, 1, 600, 500) > 1       # dirty hour: release sooner
    assert abs(carbon_factor(1, 0.0, 600, 500) - 1) < 1e-9   # zero tolerance: no change
    assert abs(carbon_factor(0, 1, 600, 500) - 1) < 1e-9     # zero weight: no change


def test_no_prewarm_is_all_cold_for_isolated_calls():
    fns, train, test = synthetic_dataset(3)
    est = MeanRatePredictor().fit_predict(train, fns)
    none = simulate(fns, test, est, "india", Policy(kind="fixed", keep=0), 3)
    long = simulate(fns, test, est, "india", Policy(kind="fixed", keep=30), 3)
    assert none.cold_rate > long.cold_rate
    assert none.gb_min < long.gb_min


def test_greenwarm_beats_fixed_10_on_default_setup():
    out = run_all("india", "mean", 7, 10, 1.0)
    fixed = out["results"]["Fixed keep-alive 10 min"]
    gw = out["results"]["GreenWarm"]
    assert gw.cold_rate <= fixed.cold_rate
    assert gw.co2_g < fixed.co2_g


def test_shifting_to_cleaner_region_cuts_emissions():
    base = run_all("india", "mean", 7, 10, 1.0)["results"]["GreenWarm"]
    shifted = run_all("india", "mean", 7, 10, 1.0, shift_to="sweden")["results"]["GreenWarm"]
    assert shifted.co2_g < base.co2_g


def test_gbm_predictor_runs():
    fns, train, test = synthetic_dataset(2)
    est = GBMRatePredictor(max_iter=40).fit_predict(train, fns)
    assert est.shape == (len(fns), 1440) and (est >= 0).all()


def test_azure_loader_with_mock_files(tmp_path):
    rng = np.random.default_rng(0)
    files = []
    for d in range(3):
        df = pd.DataFrame({"HashOwner": "o", "HashApp": "a", "HashFunction": [f"f{i}" for i in range(5)],
                           "Trigger": ["http", "timer", "queue", "http", "http"]})
        for m in range(1, 1441):
            df[str(m)] = rng.poisson(0.2, 5)
        p = tmp_path / f"d{d}.csv"
        df.to_csv(p, index=False)
        files.append(str(p))
    specs, train, test = load_azure_trace(files, n_functions=3, min_total=10)
    assert len(specs) == 3 and len(train) == 2 and test.shape == (3, 1440)
