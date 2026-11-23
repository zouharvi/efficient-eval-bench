import math
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import toml

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../")

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "stix"

RANDOM_METHOD = "Random (uniform)"
METRICS: list[tuple[str, str, str, bool]] = [
    ("corr_tau", "Model ranking", r"model ranking $\tau$", False),
    ("corr_wtau", "Model ranking", r"weighted model ranking $\tau$", False),
    ("mae_model", "Model-level", "model avg score MAE", True),
    ("mae_item", "Item-level", "model-item score MAE", True),
]


def _fmt_bound(v: float) -> str:
    if abs(v) < 1e-12:
        return "0"
    if abs(v - 1.0) < 1e-12:
        return "1"
    return f"{v:.2f}"


def _method_mean(method_res: dict[str, dict[str, float]], metric: str) -> float | None:
    vals = [float(e[metric]) for e in method_res.values() if metric in e and not math.isnan(float(e[metric]))]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    with open("computed/01-results_all.toml", "r") as f:
        results = toml.load(f)

    scores_by_metric: dict[str, np.ndarray] = {}
    rand_by_metric: dict[str, float] = {}
    mae_max = 0.0

    for metric, _, _, lower_is_better in METRICS:
        method_avgs = [
            avg
            for method_name, method_res in results.items()
            if method_name != RANDOM_METHOD and (avg := _method_mean(method_res, metric)) is not None
        ]
        rand = _method_mean(results.get(RANDOM_METHOD, {}), metric)
        assert rand is not None
        if lower_is_better:
            mae_max = max(mae_max, *method_avgs, rand)
        scores_by_metric[metric] = np.array(method_avgs, dtype=float)
        rand_by_metric[metric] = rand

    _, axes = plt.subplots(len(METRICS), 1, figsize=(4.5, 3.2))

    for ax, (metric, title, subtitle, lower_is_better) in zip(axes, METRICS):
        scores = scores_by_metric[metric]
        rand = rand_by_metric[metric]
        x_min, x_max = (0.0, mae_max) if lower_is_better else (0.4, 1.0)
        span = max(1e-12, x_max - x_min)

        if len(scores) > 1 and np.std(scores) > 0:
            kde = scipy.stats.gaussian_kde(scores)
            x = np.sort(np.unique(np.concatenate([np.linspace(float(scores.min()), float(scores.max()), 300), [x_min, rand, x_max]])))
            x = x[(x >= float(scores.min())) & (x <= float(scores.max()))]
            h = kde(x)
            h_max = float(np.max(h))
            h = 1.0 * h / h_max if h_max > 0 else np.zeros_like(h)

            green_mask = x <= rand if lower_is_better else x >= rand
            red_mask = x >= rand if lower_is_better else x <= rand
            ax.fill_between(x, 0, h, where=green_mask, interpolate=True, color="#2e7d32", lw=0, zorder=2)
            ax.fill_between(x, 0, h, where=red_mask, interpolate=True, color="#991b1b", lw=0, zorder=2)

            pct = 100.0 * float(np.mean(scores > rand if lower_is_better else scores < rand))
            if metric == "mae_item":
                x_text, ha = rand - 0.025 * span, "left"
            elif lower_is_better:
                x_text, ha = (rand + 0.045 * span, "right") if (x_max - rand) >= 0.25 * span else (x_max, "left")
            else:
                x_text, ha = (rand - 0.045 * span, "right") if (rand - x_min) >= 0.25 * span else (x_min, "left")
            ax.text(x_text, 0.55, f"{pct:.0f}% methods are\nworse than random", ha=ha, va="bottom", color="black", fontsize=7, zorder=5)

        ax.vlines(rand, 0, 1.0, color="black", lw=1.4, zorder=4)

        ax.set_xlim(x_min, x_max)
        if lower_is_better:
            ax.invert_xaxis()
        ax.set_ylim(0, 1.05)
        ax.set_xticks([x_min, x_max])
        ax.set_xticklabels([_fmt_bound(x_min), _fmt_bound(x_max)])
        ax.set_yticks([])

        ax.text(-0.04, 0.42, title, transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color="black")
        ax.text(-0.04, 0.34, subtitle, transform=ax.transAxes, ha="right", va="top", fontsize=7, color="black")

        for spine in ("top", "left", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_bounds(x_min, x_max)

    plt.tight_layout(pad=0.0)
    plt.savefig("computed/03-dist_global.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
