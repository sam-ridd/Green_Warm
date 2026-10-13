"""Carbon intensity curves (gCO2 per kWh) for each region.

The hourly values below are illustrative shapes, not measurements. To use real data,
call load_carbon_csv() with a file from Electricity Maps or an India grid source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MINUTES = 1440

REGIONS: dict[str, dict] = {
    "india": {
        "name": "India (national grid)",
        "hourly": [720, 725, 730, 725, 715, 700, 680, 640, 600, 560, 520, 500,
                   500, 505, 520, 560, 620, 690, 740, 770, 780, 765, 745, 730],
    },
    "germany": {
        "name": "Germany",
        "hourly": [380, 375, 370, 370, 375, 385, 390, 370, 330, 290, 260, 245,
                   240, 245, 260, 290, 340, 400, 440, 450, 440, 425, 405, 390],
    },
    "california": {
        "name": "California",
        "hourly": [220, 215, 210, 210, 215, 225, 230, 200, 150, 110, 85, 75,
                   70, 70, 75, 90, 130, 210, 290, 310, 300, 270, 245, 230],
    },
    "sweden": {
        "name": "Sweden",
        "hourly": [38, 36, 35, 35, 36, 38, 42, 48, 52, 50, 46, 42,
                   40, 40, 42, 45, 50, 55, 54, 50, 46, 43, 41, 40],
    },
}


def hourly_to_minutes(hourly) -> np.ndarray:
    """Linearly interpolate 24 hourly values to 1440 per-minute values (wraps at midnight)."""
    h = np.asarray(hourly, dtype=float)
    x = np.arange(MINUTES) / 60.0
    i = np.floor(x).astype(int) % 24
    j = (i + 1) % 24
    f = x - np.floor(x)
    return h[i] * (1 - f) + h[j] * f


def ci_series(region: str) -> np.ndarray:
    return hourly_to_minutes(REGIONS[region]["hourly"])


def load_carbon_csv(path: str, hour_col: str = "hour", value_col: str = "gco2_per_kwh") -> np.ndarray:
    """Load a 24-row CSV (hour 0-23, gCO2/kWh) and return a per-minute series."""
    df = pd.read_csv(path).sort_values(hour_col)
    if len(df) != 24:
        raise ValueError("Expected 24 hourly rows; average your data by hour of day first.")
    return hourly_to_minutes(df[value_col].to_numpy())
