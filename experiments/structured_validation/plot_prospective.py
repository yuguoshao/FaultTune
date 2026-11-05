"""Render the complete prespecified comparison; never analyze partial results."""
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT/"results/structured_validation/prospective_v1"


def label(method):
    return method.replace("weighted_conditional_unequal", "unequal Fisher").replace("pair_grid", "paired grid").replace("_", " ")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    analysis = json.loads((DIRECTORY/"analysis.json").read_text(encoding="utf-8"))
    summary = json.loads((DIRECTORY/"summary.json").read_text(encoding="utf-8"))
    audit = json.loads((DIRECTORY/"audit.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and audit["status"] == "passed"
    assert set(audit["input_sha256"]) == {"spec.json", "summary.json", "trials.jsonl", "analysis.json", "selection.json", "release.json"}
    for name, expected in audit["input_sha256"].items():
        assert digest(DIRECTORY/name) == expected, name
    primary, secondary = analysis["primary"], analysis["secondary"]
    assert audit["primary_pass"] == analysis["primary_numerical_threshold_met"] == (primary["percentile_95_ci"][1] < 1)
    rows = [primary]
    if primary["comparator"] != secondary["comparator"]:
        rows.append(secondary)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4), gridspec_kw={"width_ratios": [1.1, 1.5]})
    colors = ["#1565a2", "#8a969f"]
    for i, row in enumerate(rows):
        point = row["geometric_mean_ratio"]
        lo, hi = row["percentile_95_ci"]
        axes[0].hlines(i, lo, hi, color=colors[i], linewidth=1.5)
        axes[0].vlines([lo, hi], i-.05, i+.05, color=colors[i], linewidth=1.5)
        axes[0].scatter([point], [i], color=colors[i], s=45, zorder=3)
        axes[0].text(point, i+.15, "{:.6f}\n[{:.8f}, {:.8f}]".format(point, lo, hi),
                     ha="center", va="top", fontsize=9)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(["Primary: "+label(rows[0]["comparator"])]+([
        "Secondary: "+label(rows[1]["comparator"])] if len(rows)>1 else []))
    axes[0].set_ylim(-.5, max(1., len(rows)-.35))
    axes[0].invert_yaxis()
    axes[0].axvline(1, color="#a84739", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Candidate / comparator: geometric mean cost ratio")
    axes[0].set_title("Fixed-case aggregate, 95% bootstrap interval", loc="left", fontsize=11)
    ratios = sorted(primary["per_case_ratios"], key=lambda row: (row["ratio"], row["case_id"]))
    vals = np.array([row["ratio"] for row in ratios])
    bar_colors = np.where(vals<1, "#1565a2", np.where(vals>1, "#c36a51", "#8a969f"))
    axes[1].bar(np.arange(len(vals)), vals-1, bottom=1, color=bar_colors, width=.85)
    axes[1].axhline(1, color="#46505a", linewidth=1)
    axes[1].set_xlim(-1, len(vals))
    axes[1].set_ylim(min(.95, float(vals.min())-.06), max(1.05, float(vals.max())+.06))
    axes[1].set_xlabel("48 fixed main cases, sorted by cost ratio")
    axes[1].set_ylabel("Ratio of unconditional mean total shots")
    axes[1].set_title("Candidate versus selected primary comparator", loc="left", fontsize=11)
    axes[1].text(.02, .96, "Faster {} | tied {} | slower {}".format(
        primary["faster_cases"], primary["tied_cases"], primary["slower_cases"]),
        transform=axes[1].transAxes, va="top", fontsize=9)
    accepted = analysis["primary_numerical_threshold_met"]
    title = "FaultTune: "+label(primary["candidate"])+(" passes" if accepted else " does not pass")+" the prespecified comparison"
    fig.suptitle(title, x=.03, ha="left", fontsize=13, fontweight="bold")
    fig.text(.03, .075, "64 independent joint replicates per main case; 5,000 paired bootstrap resamples. Errors and cap abstentions included.", fontsize=9)
    fig.text(.03, .035, "Audit passed. Fixed simulator cases and named implementations only; this is not a full-field SOTA claim.", fontsize=9, color="#59616a")
    fig.tight_layout(rect=[.01, .13, .995, .91], w_pad=2.4)
    path = DIRECTORY/"comparison.png"
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)
    print(str(path))


if __name__ == "__main__":
    main()
