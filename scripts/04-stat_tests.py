# Paired tests of each method vs. random sampling, with the data seed as the unit.

import argparse
import collections
import itertools
import json
import os

import numpy as np
import scipy.stats
import toml

args = argparse.ArgumentParser()
args.add_argument("--random", default="Random (uniform)", help="Baseline method.")
args.add_argument("--budgets", type=float, nargs="+", default=None, help="Budget fractions to average over. Default: all except 1.0 (where every method sees all data).")
args.add_argument("--alpha", type=float, default=0.05)
args.add_argument("--n-boot", type=int, default=10_000)
args = args.parse_args()

os.chdir(os.path.dirname(__file__) + "/../")

METRICS = ["corr_tau", "corr_wtau", "mae_model", "mae_item"]
SIGN = {"corr_tau": 1, "corr_wtau": 1, "mae_model": -1, "mae_item": -1}
# (base, strongest) scenario of each failure mode
TRENDS = {
    "model count": ("256 models", "4 models"),
    "item heteroscedasticity": ("base", "Item heteroscedasticity 3"),
    "model heteroscedasticity": ("base", "Model heteroscedasticity 3"),
    "metric noise": ("Metric noise 0.5", "Metric noise 2.0"),
    "annotator strict": ("base", "Annotator strict 3"),
    "annotator inconsistency": ("base", "Annotator inconsistency 3"),
    "annotator replacement": ("base", "Annotator replacement 3"),
}
RNG = np.random.default_rng(0)


with open("computed/01-results_raw.jsonl") as f:
    rows = [json.loads(line) for line in f if line.strip()]

budgets = set(args.budgets) if args.budgets else {r["budget_frac"] for r in rows} - {1.0}
budgets = {round(b, 4) for b in budgets}
acc = collections.defaultdict(list)
for r in rows:
    if r["budget_frac"] in budgets:
        acc[(r["method"], r["scenario"], r["data_seed"])].append(r)

SEEDS = sorted({k[2] for k in acc})
METHODS = sorted({k[0] for k in acc} - {args.random})
SCENARIOS = sorted({k[1] for k in acc})


def values(method, scenario, metric):
    out = []
    for s in SEEDS:
        vs = [r[metric] for r in acc.get((method, scenario, s), []) if r[metric] is not None and not np.isnan(r[metric])]
        out.append(np.mean(vs) if vs else np.nan)
    return np.array(out)


def diff(method, scenario, metric):
    # positive = method better than random
    return SIGN[metric] * (values(method, scenario, metric) - values(args.random, scenario, metric))


def signflip_p(d):
    # paired sign-flip permutation test, exact for <= 16 seeds
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2 or np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    if n <= 16:
        signs = np.array(list(itertools.product([1, -1], repeat=n)))
    else:
        signs = RNG.choice([1, -1], size=(args.n_boot, n))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-12) + (n > 16)) / (len(signs) + (n > 16)))


def boot_ci(d):
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return (float("nan"), float("nan"))
    means = d[RNG.integers(0, len(d), size=(args.n_boot, len(d)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [args.alpha / 2, 1 - args.alpha / 2]))


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    adj[order] = np.minimum(1, np.maximum.accumulate(ps[order] * (len(ps) - np.arange(len(ps)))))
    return adj


def bh(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    ranked = ps[order] * len(ps) / (np.arange(len(ps)) + 1)
    adj = np.empty_like(ps)
    adj[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return adj


def summarize(d, p, p_adj):
    lo, hi = boot_ci(d)
    return {
        "diff": float(np.nanmean(d)),
        "ci_low": lo,
        "ci_high": hi,
        "p": p,
        "p_adj": float(p_adj),
        "significant": bool(p_adj < args.alpha),
        "n_seeds": int(np.sum(~np.isnan(d))),
    }


min_p = 2 / 2 ** len(SEEDS)
print(f"{len(METHODS)} methods, {len(SCENARIOS)} scenarios, {len(SEEDS)} data seeds, budgets {sorted(budgets)}")
if min_p * len(METHODS) > args.alpha:
    print(f"WARNING: with {len(SEEDS)} seeds the smallest attainable p is {min_p:.4f}; after Holm over {len(METHODS)} methods nothing can be significant. Use more --data-seeds (>= {int(np.ceil(np.log2(2 * len(METHODS) / args.alpha)))}).")

out = {"meta": {"random": args.random, "seeds": len(SEEDS), "budgets": sorted(budgets), "alpha": args.alpha}, "overall": {}, "scenario": {}, "trend": {}}

# overall
for metric in METRICS:
    ds = {}
    for m in METHODS:
        per_scen = np.array([diff(m, sc, metric) for sc in SCENARIOS])
        per_scen = per_scen[~np.all(np.isnan(per_scen), axis=1)]
        ds[m] = np.nanmean(per_scen, axis=0) if len(per_scen) else np.full(len(SEEDS), np.nan)
    ps = [signflip_p(ds[m]) for m in METHODS]
    for m, p, pa in zip(METHODS, ps, holm(ps)):
        out["overall"].setdefault(m, {})[metric] = summarize(ds[m], p, pa)

for metric in METRICS:
    keys, ds = [], []
    for m in METHODS:
        for sc in SCENARIOS:
            d = diff(m, sc, metric)
            if np.sum(~np.isnan(d)) >= 2:
                keys.append((m, sc))
                ds.append(d)
    ps = [signflip_p(d) for d in ds]
    for (m, sc), d, p, pa in zip(keys, ds, ps, bh(ps)):
        out["scenario"].setdefault(m, {}).setdefault(sc, {})[metric] = summarize(d, p, pa)

# trends: negative = method degrades more than random
for family, (base, strong) in TRENDS.items():
    if base not in SCENARIOS or strong not in SCENARIOS:
        continue
    for metric in METRICS:
        r_base, r_strong = values(args.random, base, metric), values(args.random, strong, metric)
        random_change = SIGN[metric] * (r_strong - r_base)
        ms, ds = [], []
        for m in METHODS:
            d = diff(m, strong, metric) - diff(m, base, metric)
            if np.sum(~np.isnan(d)) >= 2:
                ms.append(m)
                ds.append(d)
        ps = [signflip_p(d) for d in ds]
        for m, d, p, pa in zip(ms, ds, ps, bh(ps)):
            entry = summarize(d, p, pa)
            entry["method_change"] = float(SIGN[metric] * np.nanmean(values(m, strong, metric) - values(m, base, metric)))
            entry["random_change"] = float(np.nanmean(random_change))
            out["trend"].setdefault(family, {}).setdefault(m, {})[metric] = entry

# correlation between objectives across methods
obj = np.array([[out["overall"][m][metric]["diff"] for metric in METRICS] for m in METHODS])
obj = obj[~np.isnan(obj).any(axis=1)]
out["objectives"] = {}
for (i, a), (j, b) in itertools.combinations(enumerate(METRICS), 2):
    rho = scipy.stats.spearmanr(obj[:, i], obj[:, j]).statistic
    boots = []
    for _ in range(2000):
        idx = RNG.integers(0, len(obj), len(obj))
        boots.append(scipy.stats.spearmanr(obj[idx, i], obj[idx, j]).statistic)
    lo, hi = np.nanquantile(boots, [args.alpha / 2, 1 - args.alpha / 2])
    out["objectives"][f"{a} vs {b}"] = {"rho": float(rho), "ci_low": float(lo), "ci_high": float(hi)}

with open("computed/04-stat_tests.toml", "w") as f:
    toml.dump(out, f)

print(f"\nOverall vs. {args.random} (Holm, alpha={args.alpha}): significantly better / worse / n.s.")
for metric in METRICS:
    res = [out["overall"][m][metric] for m in METHODS]
    better = sum(r["significant"] and r["diff"] > 0 for r in res)
    worse = sum(r["significant"] and r["diff"] < 0 for r in res)
    print(f"  {metric:10s} {better:3d} / {worse:3d} / {len(res) - better - worse:3d}")
print("\nTrends: methods degrading significantly more than random (BH)")
for family, per_m in out["trend"].items():
    counts = {metric: sum(v[metric]["significant"] and v[metric]["diff"] < 0 for v in per_m.values() if metric in v) for metric in METRICS}
    print(f"  {family:26s} " + "  ".join(f"{k}={v}" for k, v in counts.items()))
print("\nSpearman between objectives (over methods, bootstrap CI)")
for k, v in out["objectives"].items():
    print(f"  {k:24s} {v['rho']:+.2f} [{v['ci_low']:+.2f}, {v['ci_high']:+.2f}]")
