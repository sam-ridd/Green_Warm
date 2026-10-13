"""Run the baselines and GreenWarm on one dataset and return tidy results."""
from __future__ import annotations

import pandas as pd

from .predictor import PREDICTORS, evaluate_predictor
from .simulator import Policy, simulate
from .traffic import synthetic_dataset


def run_all(region: str = "india", predictor: str = "mean", seed: int = 7, keep: int = 10,
            carbon_weight: float = 1.0, shift_to: str | None = None, dataset=None):
    functions, train, test = dataset if dataset is not None else synthetic_dataset(seed)
    est = PREDICTORS[predictor]().fit_predict(train, functions)
    runs = {
        "No pre-warming": Policy(kind="fixed", keep=0),
        f"Fixed keep-alive {keep} min": Policy(kind="fixed", keep=keep),
        "GreenWarm": Policy(kind="green", carbon_weight=carbon_weight, shift_to=shift_to),
        "GreenWarm (carbon off)": Policy(kind="green", carbon_weight=0.0, shift_to=None),
    }
    results = {name: simulate(functions, test, est, region, pol, seed) for name, pol in runs.items()}
    table = pd.DataFrame([{
        "policy": name,
        "cold_start_%": round(r.cold_rate * 100, 2),
        "p95_latency_ms": round(r.p95_ms, 1),
        "warm_GB_min": round(r.gb_min, 0),
        "emissions_g": round(r.co2_g, 2),
        "prewarm_hit_%": None if r.hit_rate is None else round(r.hit_rate * 100, 1),
    } for name, r in results.items()])
    return {"functions": functions, "est": est, "test": test, "results": results,
            "table": table, "predictor_quality": evaluate_predictor(est, test)}


def tradeoff_sweep(functions, test, est, region, seed=7, shift_to=None):
    fixed = []
    for k in (0, 1, 2, 3, 5, 10, 15, 20, 30):
        r = simulate(functions, test, est, region, Policy(kind="fixed", keep=k), seed)
        fixed.append((k, r.cold_rate * 100, r.co2_g))
    green = []
    for w in (0, 0.5, 1, 1.5, 2, 2.5, 3):
        r = simulate(functions, test, est, region, Policy(kind="green", carbon_weight=w, shift_to=shift_to), seed)
        green.append((w, r.cold_rate * 100, r.co2_g))
    return pd.DataFrame(fixed, columns=["keep_min", "cold_%", "co2_g"]), pd.DataFrame(green, columns=["carbon_weight", "cold_%", "co2_g"])
