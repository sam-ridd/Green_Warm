"""Invocation traffic: a synthetic generator and a loader for the Azure Functions public trace."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

MINUTES = 1440


def _bump(h: float, centre: float, width: float) -> float:
    d = abs(h - centre)
    d = min(d, 24 - d)
    return float(np.exp(-(d * d) / (2 * width * width)))


@dataclass
class FunctionSpec:
    name: str
    pattern: str
    sla: str              # strict | standard | tolerant
    tolerance: float      # how much latency may be traded for carbon (0 to 1)
    mem_gb: float         # memory held while warm
    cold_ms: float        # typical cold-start time
    smooth: int           # +/- minutes the empirical predictor averages over
    rate: Callable[[float], float] | None = None   # expected calls per minute at hour h


SYNTHETIC_FUNCTIONS = [
    FunctionSpec("checkout-api", "Shopper traffic, peaks near noon and 8 pm", "strict", 0.15, 0.5, 850, 2,
                 lambda h: 1.6 * (_bump(h, 12.5, 2.5) + 0.8 * _bump(h, 20.5, 1.8)) + 0.03),
    FunctionSpec("sensor-ingest", "Device pings exactly every 15 minutes", "standard", 0.5, 0.25, 450, 0,
                 lambda h: 0.95 if round(h * 60) % 15 == 0 else 0.01),
    FunctionSpec("notify-user", "Push messages, evening burst plus a morning one", "standard", 0.5, 0.35, 500, 2,
                 lambda h: 0.6 * _bump(h, 19, 1.5) + 0.12 * _bump(h, 8, 1) + 0.02),
    FunctionSpec("image-resize", "Uploads during the day", "tolerant", 1.0, 0.75, 1100, 2,
                 lambda h: 1.0 * _bump(h, 15, 4) + 0.02),
    FunctionSpec("report-builder", "Scheduled reports at 9 am and 5 pm", "tolerant", 1.0, 1.0, 1700, 1,
                 lambda h: 0.7 * (_bump(h, 9, 0.4) + _bump(h, 17, 0.4)) + 0.01),
    FunctionSpec("nightly-etl", "Batch loads between 1 am and 3 am", "tolerant", 1.0, 1.5, 2200, 2,
                 lambda h: 0.5 * _bump(h, 2, 0.6) + 0.005),
]


def generate_day(seed: int, functions=SYNTHETIC_FUNCTIONS) -> np.ndarray:
    """One day of per-minute call counts, shape (n_functions, 1440)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(functions), MINUTES), dtype=np.int32)
    for fi, f in enumerate(functions):
        lam = np.array([f.rate(m / 60.0) for m in range(MINUTES)])
        out[fi] = rng.poisson(lam)
    return out


def synthetic_dataset(seed: int = 7, train_days: int = 6):
    """Returns (functions, train, test) where train is a list of day arrays."""
    train = [generate_day(seed * 1000 + d) for d in range(train_days)]
    test = generate_day(seed * 1000 + 99)
    return SYNTHETIC_FUNCTIONS, train, test


def load_azure_trace(files: list[str], n_functions: int = 12, min_total: int = 200):
    """Load Azure Functions 2019 trace day files (invocations_per_function_md.anon.dNN.csv).

    Each file has columns HashOwner, HashApp, HashFunction, Trigger, then '1'..'1440'
    (calls in each minute). The last file is the test day, the earlier ones train the predictor.
    Picks the busiest functions that appear in every file. Memory and cold-start time are not in
    this file, so defaults are used (override FunctionSpec fields if you merge the memory trace).
    """
    if len(files) < 2:
        raise ValueError("Give at least two day files: one or more for training and one for testing.")
    frames = [pd.read_csv(p).set_index("HashFunction") for p in files]
    keep = set(frames[0].index)
    for fr in frames[1:]:
        keep &= set(fr.index)
    minute_cols = [str(i) for i in range(1, MINUTES + 1)]
    totals = {k: sum(fr.loc[k, minute_cols].to_numpy().sum() for fr in frames) for k in keep}
    ranked = [k for k, v in sorted(totals.items(), key=lambda kv: -kv[1]) if v >= min_total][:n_functions]
    if not ranked:
        raise ValueError("No functions with enough calls; lower min_total.")
    days = [np.stack([fr.loc[k, minute_cols].to_numpy().astype(np.int32) for k in ranked]) for fr in frames]
    specs = []
    for i, k in enumerate(ranked):
        trig = frames[0].loc[k, "Trigger"] if "Trigger" in frames[0].columns else "http"
        tolerant = str(trig).lower() in {"queue", "timer", "storage", "eventhub", "orchestration", "others"}
        specs.append(FunctionSpec(f"fn-{str(k)[:6]}", f"Azure trace, trigger: {trig}",
                                  "tolerant" if tolerant else "strict",
                                  1.0 if tolerant else 0.15, 0.5, 1000, 2))
    return specs, days[:-1], days[-1]
