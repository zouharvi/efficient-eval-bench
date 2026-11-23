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
METRIC = "mae_item"

CATEGORIES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "models": (
        "Number of models",
        [
            ("4", "4 models"),
            ("16", "16 models"),
            ("64", "64 models"),
            ("256", "256 models"),
        ],
    ),
    "model_hetero": (
        "Model heteroscedasticity",
        [
            ("none", "base"),
            ("1", "Model heteroscedasticity 1"),
            ("2", "Model heteroscedasticity 2"),
            ("3", "Model heteroscedasticity 3"),
        ],
    ),
    "item_hetero": (
        "Item heteroscedasticity",
        [
            ("none", "base"),
            ("1", "Item heteroscedasticity 1"),
            ("2", "Item heteroscedasticity 2"),
            ("3", "Item heteroscedasticity 3"),
        ],
    ),
    "metric_noise": (
        "Metric noise",
        [
            ("none", "base"),
            ("0.5", "Metric noise 0.5"),
            ("1.0", "Metric noise 1.0"),
            ("2.0", "Metric noise 2.0"),
        ],
    ),
    "annot_strict": (
        "Annotator strictness",
        [
            ("none", "base"),
            ("1", "Annotator strict 1"),
            ("2", "Annotator strict 2"),
            ("3", "Annotator strict 3"),
        ],
    ),
    "annot_inconsistency": (
        "Annotator inconsistency",
        [
            ("none", "base"),
            ("1", "Annotator inconsistency 1"),
            ("2", "Annotator inconsistency 2"),
            ("3", "Annotator inconsistency 3"),
        ],
    ),
    "annot_replacement": (
        "Annotator replacement",
        [
            ("none", "base"),
            ("1", "Annotator replacement 1"),
            ("2", "Annotator replacement 2"),
            ("3", "Annotator replacement 3"),
        ],
    ),
}


def main() -> None:
    with open("computed/01-results_all.toml", "r") as f:
        results = toml.load(f)

    for cat_key, (xlabel, items) in CATEGORIES.items():
        _, ax = plt.subplots(figsize=(3.5, 2.5))
        xs = list(range(len(items)))

        for pos, (_, data_name) in enumerate(items):
            rand = float(results[RANDOM_METHOD][data_name][METRIC])

            scores = np.array(
                [
                    float(vals[data_name][METRIC])
                    for method_name, vals in results.items()
                    if method_name != RANDOM_METHOD and data_name in vals and not math.isnan(float(vals[data_name][METRIC]))
                ]
            )
            if len(scores) == 0:
                continue

            if len(scores) > 1 and np.std(scores) > 0:
                kde = scipy.stats.gaussian_kde(scores)
                y = np.sort(np.unique(np.append(np.linspace(scores.min(), scores.max(), 300), rand)))
                y = y[(y >= scores.min()) & (y <= scores.max())]
                w = kde(y)
                w_max = float(np.max(w))
                w = 0.5 * w / w_max if w_max > 0 else np.zeros_like(w)

                ax.fill_betweenx(y, pos - w, pos, where=y <= rand, interpolate=True, color="#2e7d32", lw=0, zorder=2) # type: ignore
                ax.fill_betweenx(y, pos - w, pos, where=y >= rand, interpolate=True, color="#991b1b", lw=0, zorder=2) # type: ignore

                pct = 100.0 * float(np.mean(scores > rand))
                y_text = rand
                va = "top"
                ax.text(pos + 0.05, y_text, f"{pct:.0f}%\nworse\nthan\nrandom", ha="left", va=va, color="black", fontsize=7, zorder=5)

            ax.hlines(rand, pos - 0.5, pos + 0.04, color="black", lw=1.4, zorder=4)

        ax.set_xticks(xs)
        ax.set_xticklabels([label for label, _ in items])
        ax.set_xlim(-0.5, len(items) - 0.3)
        ax.margins(y=0.08)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Item MAE")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout(pad=0.1)
        plt.savefig(f"computed/02-dist_{cat_key}.svg")
        


if __name__ == "__main__":
    main()
