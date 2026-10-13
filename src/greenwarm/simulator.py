"""Minute-by-minute serverless container simulator with three policies.

* fixed keep-alive (keep = 0 means no pre-warming at all)
* GreenWarm: predictor-driven pre-warm and keep-alive whose thresholds and look-ahead
  window move with the carbon intensity of the grid and the function's latency tolerance.

One container per function. A container that is warm serves calls at warm latency; otherwise
the first call in that minute is a cold start.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .carbon import ci_series
from .predictor import prefix_sums

MINUTES = 1440
W_PER_GB = 3.0        # assumed watts drawn per GB of memory held warm
WARM_MS = 28.0        # typical warm invocation latency
SHIFT_MS = 80.0       # extra round trip when a function runs in another region


@dataclass
class Policy:
    kind: str = "green"            # "fixed" or "green"
    keep: int = 10                 # fixed keep-alive minutes (fixed only)
    carbon_weight: float = 1.0     # lambda: 0 ignores carbon
    theta_pre: float = 0.5         # pre-warm if P(call in next 2 min) >= theta_pre * cf
    theta_keep: float = 0.2        # keep warm if P(call in next W/cf min) >= theta_keep * cf
    window: int = 5                # base look-ahead window W in minutes
    shift_to: str | None = None    # region where tolerant functions run


@dataclass
class Result:
    invocations: int = 0
    cold: int = 0
    gb_min: float = 0.0
    co2_g: float = 0.0
    prewarms: int = 0
    prewarm_hits: int = 0
    p95_ms: float = 0.0
    hour_gb_min: np.ndarray = field(default_factory=lambda: np.zeros(24))
    per_function: list = field(default_factory=list)

    @property
    def cold_rate(self) -> float:
        return self.cold / self.invocations if self.invocations else 0.0

    @property
    def hit_rate(self):
        return self.prewarm_hits / self.prewarms if self.prewarms else None


def carbon_factor(weight: float, tolerance: float, ci_now: float, ci_mean: float) -> float:
    """cf < 1 on clean hours, cf > 1 on dirty hours; strict functions barely move."""
    return float(np.clip(np.exp(2.0 * weight * tolerance * (ci_now / ci_mean - 1.0)), 0.3, 3.0))


def simulate(functions, test_day: np.ndarray, est: np.ndarray, region: str, policy: Policy, seed: int = 1) -> Result:
    rng = np.random.default_rng(seed * 7919 + 13)
    pre_all = prefix_sums(est)
    ci_home = ci_series(region)
    res = Result()
    lat_all: list[float] = []

    for fi, f in enumerate(functions):
        cnt, pre = test_day[fi], pre_all[fi]
        shifted = policy.kind == "green" and policy.shift_to is not None and f.sla == "tolerant"
        ci = ci_series(policy.shift_to) if shifted else ci_home
        ci_mean = ci.mean()
        extra = SHIFT_MS if shifted else 0.0
        kwh_per_min = f.mem_gb * W_PER_GB / 60 / 1000

        warm, last_call = False, -10**9
        r = dict(name=f.name, inv=0, cold=0, gb_min=0.0, co2=0.0, prewarms=0, hits=0)

        for m in range(MINUTES):
            if policy.kind == "green" and not warm:
                cf = carbon_factor(policy.carbon_weight, f.tolerance, ci[m], ci_mean)
                p = 1 - np.exp(-(pre[m + 2] - pre[m]))
                if p >= min(0.95, policy.theta_pre * cf):
                    warm = True
                    r["prewarms"] += 1
                    if cnt[m] + cnt[(m + 1) % MINUTES] + cnt[(m + 2) % MINUTES] > 0:
                        r["hits"] += 1

            c = int(cnt[m])
            if c > 0:
                for k in range(c):
                    is_cold = (not warm) and k == 0
                    base = f.cold_ms * (0.85 + 0.3 * rng.random()) if is_cold else WARM_MS * (0.8 + 0.4 * rng.random())
                    lat_all.append(base + extra)
                    r["cold"] += int(is_cold)
                r["inv"] += c
                warm, last_call = True, m

            if warm:
                r["gb_min"] += f.mem_gb
                r["co2"] += kwh_per_min * ci[m]
                res.hour_gb_min[m // 60] += f.mem_gb

            if policy.kind == "fixed":
                warm = last_call >= 0 and (m + 1 - last_call) <= policy.keep
            elif warm:
                cf = carbon_factor(policy.carbon_weight, f.tolerance, ci[(m + 1) % MINUTES], ci_mean)
                w = int(np.clip(round(policy.window / cf), 1, 60))
                p = 1 - np.exp(-(pre[m + 1 + w] - pre[m + 1]))
                warm = p >= min(0.95, policy.theta_keep * cf)

        res.invocations += r["inv"]; res.cold += r["cold"]
        res.gb_min += r["gb_min"]; res.co2_g += r["co2"]
        res.prewarms += r["prewarms"]; res.prewarm_hits += r["hits"]
        res.per_function.append(r)

    if lat_all:
        res.p95_ms = float(np.percentile(lat_all, 95))
    return res
