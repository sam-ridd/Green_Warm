"""Command-line run: prints the comparison table and saves CSVs and charts to results/.

Examples:
    python run_experiment.py
    python run_experiment.py --region california --predictor gbm --carbon-weight 1.5
    python run_experiment.py --azure data/d01.csv data/d02.csv data/d03.csv   # last file = test day
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from greenwarm.carbon import REGIONS, ci_series
from greenwarm.experiment import run_all, tradeoff_sweep
from greenwarm.traffic import load_azure_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="india", choices=list(REGIONS))
    ap.add_argument("--predictor", default="mean", choices=["mean", "gbm"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep", type=int, default=10, help="fixed keep-alive baseline in minutes")
    ap.add_argument("--carbon-weight", type=float, default=1.0)
    ap.add_argument("--shift-to", default=None, choices=[None] + list(REGIONS))
    ap.add_argument("--azure", nargs="*", help="Azure trace day files; the last one is the test day")
    args = ap.parse_args()

    dataset = None
    if args.azure:
        dataset = load_azure_trace(args.azure)

    out = run_all(args.region, args.predictor, args.seed, args.keep, args.carbon_weight, args.shift_to, dataset)
    print("\nPredictor quality on the test day:", {k: round(v, 3) for k, v in out["predictor_quality"].items()})
    print("\n", out["table"].to_string(index=False), "\n")

    res = Path(__file__).parent / "results"
    res.mkdir(exist_ok=True)
    out["table"].to_csv(res / "summary.csv", index=False)

    fixed, green = tradeoff_sweep(out["functions"], out["test"], out["est"], args.region, args.seed, args.shift_to)
    fixed.to_csv(res / "sweep_fixed.csv", index=False)
    green.to_csv(res / "sweep_green.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(fixed["cold_%"], fixed["co2_g"], "o-", color="#4f6d8f", label="Fixed keep-alive")
    ax.plot(green["cold_%"], green["co2_g"], "o-", color="#14805a", label="GreenWarm")
    ax.set_xlabel("Cold-start rate (%)"); ax.set_ylabel("Emissions from warm containers (g)")
    ax.set_title(f"Latency vs carbon ({args.region}, {args.predictor} predictor)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(res / "tradeoff.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4))
    hours = range(24)
    ax.bar([h - 0.2 for h in hours], out["results"][f"Fixed keep-alive {args.keep} min"].hour_gb_min, 0.4, color="#4f6d8f", label="Fixed")
    ax.bar([h + 0.2 for h in hours], out["results"]["GreenWarm"].hour_gb_min, 0.4, color="#14805a", label="GreenWarm")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Warm GB-min per hour")
    ax2 = ax.twinx()
    ci = ci_series(args.region).reshape(24, 60).mean(axis=1)
    ax2.plot(hours, ci, color="#c9741a", lw=2, label="Carbon intensity"); ax2.set_ylabel("gCO2/kWh")
    ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(res / "day.png", dpi=150)
    print("Saved results/summary.csv, sweep_*.csv, tradeoff.png, day.png")


if __name__ == "__main__":
    main()
