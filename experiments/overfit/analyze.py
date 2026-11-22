# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Compare overfit.py runs: python analyze.py RUN_DIR [RUN_DIR ...] --out OUT_DIR.

Reads <run>/history.json (written by overfit.py), writes plots and summary.md.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

# Same values as the OM4 DataLayout.depth_levels in src/samudra/constants.py
# (hardcoded to avoid importing torch).
DEPTHS = [2.5, 10, 22.5, 40, 65, 105, 165, 250, 375, 550, 775, 1050, 1400, 1850]
DEPTHS += [2400, 3100, 4000, 5000, 6000]
VARS = ["uo", "vo", "thetao", "so"]
ALL_VARS = [*VARS, "zos"]
# Okabe-Ito colorblind-safe palette.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]


def load(run_dir: Path) -> dict:
    path = run_dir / "history.json"
    # history.json may be rewritten while a run is live; retry partial writes.
    for attempt in range(5):
        try:
            data = json.loads(path.read_text())
            break
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(2)
    channels = data["channels"]
    if len(channels) != 77 or channels[-1] != "zos":
        raise ValueError(f"{path}: unexpected channels {channels[:2]}...")
    if not data["history"]:
        raise ValueError(f"{path}: empty history")
    for rec in data["history"]:
        if len(rec["train_ratio_per_channel"]) != len(channels):
            raise ValueError(f"{path}: bad ratio length at step {rec['step']}")
    return data


def var_ratios(ratio: list[float], var: str) -> list[float]:
    if var == "zos":
        return [ratio[76]]
    i = VARS.index(var)
    return ratio[i * 19 : (i + 1) * 19]


def stats(data: dict, rec: dict) -> dict:
    ratio = rec["train_ratio_per_channel"]
    worst = max(range(len(ratio)), key=ratio.__getitem__)
    out = {
        "step": rec["step"],
        "s_per_step": rec["train_s"] / rec["step"] if rec["step"] else float("nan"),
        "median": statistics.median(ratio),
        "max": f"{ratio[worst]:.3g} ({data['channels'][worst]})",
        "frac": sum(r < 1 for r in ratio) / len(ratio),
        "heldout": rec["heldout"]["median_ratio"],
    }
    for v in ALL_VARS:
        out[v] = statistics.median(var_ratios(ratio, v))
    return out


def table(rows: dict[str, dict]) -> str:
    head = ["run", "step", "train_s/step", "median", "max (channel)", "frac<1"]
    head += [f"med {v}" for v in ALL_VARS] + ["heldout median"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, s in rows.items():
        cells = [name, str(s["step"]), f"{s['s_per_step']:.1f}", f"{s['median']:.3g}"]
        cells += [s["max"], f"{s['frac']:.2f}"]
        cells += [f"{s[v]:.3g}" for v in ALL_VARS] + [f"{s['heldout']:.3g}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def line_plot(runs, colors, fn, ylabel, title, log=False, hline=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in runs.items():
        steps = [r["step"] for r in d["history"]]
        ax.plot(steps, [fn(r) for r in d["history"]], label=name, color=colors[name])
    if log:
        ax.set_yscale("log")
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=1)
    ax.set(xlabel="training step", ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def depth_plot(runs, colors, recs, title):
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.5), sharey=False)
    for ax, v in zip(axes[:4], VARS):
        for name in runs:
            r = var_ratios(recs[name]["train_ratio_per_channel"], v)
            ax.plot(r, DEPTHS, "o-", ms=3, label=name, color=colors[name])
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.axvline(1, color="k", ls="--", lw=1)
        ax.set(title=v, xlabel="model / persistence RMSE")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("depth (m)")
    ax = axes[4]  # zos has no depth: one marker per run
    for i, name in enumerate(runs):
        ratio = recs[name]["train_ratio_per_channel"][76]
        ax.plot(ratio, i, "o", label=name, color=colors[name])
    ax.set_xscale("log")
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.set(title="zos (surface)", xlabel="model / persistence RMSE", yticks=[])
    ax.grid(alpha=0.3)
    ax.xaxis.set_minor_formatter(NullFormatter())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=120)
    plt.close(fig)
    if path.stat().st_size > 250_000:
        raise RuntimeError(f"{path} is {path.stat().st_size} bytes (> 250 KB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = {d.name: load(d) for d in args.run_dirs}
    if len(runs) != len(args.run_dirs):
        raise ValueError("run directory names must be unique")
    colors = {n: COLORS[i % len(COLORS)] for i, n in enumerate(runs)}

    common = set.intersection(
        *({r["step"] for r in d["history"]} for d in runs.values())
    )
    if not common:
        raise ValueError("runs share no recorded step")
    matched = max(common)
    final = {n: d["history"][-1] for n, d in runs.items()}
    at_matched = {
        n: next(r for r in d["history"] if r["step"] == matched)
        for n, d in runs.items()
    }

    save(
        line_plot(
            runs,
            colors,
            lambda r: statistics.median(r["train_ratio_per_channel"]),
            "median model / persistence RMSE",
            "Median train ratio vs step (below dashed line beats persistence)",
            log=True,
            hline=1,
        ),
        args.out / "ratio_vs_step.png",
    )
    save(
        line_plot(
            runs,
            colors,
            lambda r: (
                sum(x < 1 for x in r["train_ratio_per_channel"])
                / len(r["train_ratio_per_channel"])
            ),
            "fraction of channels with ratio < 1",
            "Fraction of channels beating persistence (train sample)",
        ),
        args.out / "frac_beat_vs_step.png",
    )
    save(
        depth_plot(
            runs, colors, final, "Train ratio by depth at each run's final step"
        ),
        args.out / "ratio_by_depth_final.png",
    )
    save(
        depth_plot(
            runs, colors, at_matched, f"Train ratio by depth at matched step {matched}"
        ),
        args.out / "ratio_by_depth_matched_step.png",
    )

    final_rows = {n: stats(runs[n], final[n]) for n in runs}
    matched_rows = {n: stats(runs[n], at_matched[n]) for n in runs}
    md = (
        "# Overfit run summary\n\n"
        "Ratios are model RMSE / persistence RMSE on the memorized train sample "
        "(heldout column: heldout median ratio).\n\n"
        "## Final recorded step per run\n\n" + table(final_rows) + "\n\n"
        f"## Matched step ({matched})\n\n" + table(matched_rows) + "\n"
    )
    (args.out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
