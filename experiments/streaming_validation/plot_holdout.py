"""Render the frozen holdout outcome without changing any analysis."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/streaming_validation"


def main():
    analysis_path = RESULTS / "holdout_v1/analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    comparisons = analysis["comparisons"]
    keys = ["psr_beta", "guarded_grid_beta", "psr_safe"]
    labels = ["PSR + paired Beta (primary)",
              "Adaptive KL grid (secondary)",
              "PSR + safe two-sample (secondary)"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    for y, key in enumerate(keys):
        value = comparisons[key]
        point = value["geomean_cost_ratio"]
        lo, hi = value["ci95_percentile"]
        ax.errorbar(point, y, xerr=[[point-lo], [hi-point]],
                    fmt="o", color="#176b87" if y == 0 else "#657382",
                    capsize=4, markersize=6, linewidth=1.7)
    ax.axvline(1, color="#aa4d44", linestyle="--", linewidth=1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_ylim(2.6, -.75)
    ax.set_xlim(.907, 1.008)
    ax.set_xlabel("Closed policy cost / comparator cost")
    ax.set_title("Independent validation: primary criterion not met",
                 loc="left", fontsize=11)
    ax.text(.02, .03, "Primary 95% CI: [0.984476, 1.000045]\n"
            "All intervals concern 48 fixed cases.",
            transform=ax.transAxes, fontsize=9, va="bottom")
    ax.grid(axis="x", alpha=.15)
    values = sorted(cell["primary_over_comparator"]["psr_beta"]
                    for cell in analysis["main_case_results"])
    ax = axes[1]
    colors = ["#176b87" if value < 1 else "#aa4d44" if value > 1
              else "#87929d" for value in values]
    ax.scatter(np.arange(1, 49), values, c=colors, s=23)
    ax.axhline(1, color="#aa4d44", linestyle="--", linewidth=1)
    ax.set_xlim(0, 49)
    ax.set_ylim(.73, 1.15)
    ax.set_xlabel("Main cases sorted by cost ratio")
    ax.set_ylabel("Closed policy / PSR + paired Beta")
    ax.set_title("10 lower cost / 27 tied / 11 higher cost",
                 loc="left", fontsize=11)
    ax.grid(axis="y", alpha=.15)
    fig.suptitle("FaultTune: paid exploration and evidence reuse",
                 fontsize=14, x=.05, ha="left")
    fig.text(.05, .025, "28,672 new trajectories; 128 independent joint replicates per case; "
             "5,000 paired bootstrap draws. Secondary intervals are unadjusted.",
             fontsize=8, color="#4a525a")
    fig.tight_layout(rect=[.01, .07, 1, .92])
    output = RESULTS / "holdout_comparison.png"
    fig.savefig(str(output), dpi=180, facecolor="white")
    plt.close(fig)
    record = {"analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "primary_outcome": analysis["primary_outcome"]}
    (RESULTS / "holdout_plot_manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
