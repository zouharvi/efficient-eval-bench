import argparse
import functools
import json
import multiprocessing
import os
import statistics
import time

import numpy as np
import toml
import tqdm

import efficient_eval_bench.methods
import efficient_eval_bench.utils
from efficient_eval_bench.data import get_data_dict_all, shuffle_items

args = argparse.ArgumentParser()
args.add_argument("--no-reuse", action="store_true", help="Don't reuse results from computed/01-results_raw.jsonl even if it exists")
args.add_argument("--workers", type=int, default=20, help="Number of workers for parallelization. 0 turns off parallelization.")
args.add_argument("--data-seeds", type=int, default=5, help="Number of data seeds to generate and average across.")
args = args.parse_args()

os.chdir(os.path.dirname(__file__) + "/../")
os.makedirs("computed", exist_ok=True)

# load previous results if they happen to exist
RAW_PATH = "computed/01-results_raw.jsonl"
raw_rows: list[dict] = []
if os.path.exists(RAW_PATH) and not args.no_reuse:
    with open(RAW_PATH, "r") as f:
        for line in f:
            try:
                raw_rows.append(json.loads(line))
            except json.JSONDecodeError:
                # last line of an interrupted run
                pass
computed_keys = {(r["method"], r["scenario"], r["data_seed"]) for r in raw_rows}

# instantiate and load data
DATA_BY_SEED = [get_data_dict_all(seed=s) for s in range(args.data_seeds)]
BUDGETS_FRAC = np.linspace(0.1, 1.0, 10)

# prepare the parallelization queue
parallelization_queue = []
for method_name, method_fn in efficient_eval_bench.methods.METHODS_BUDGETS.items():
    for data_seed, DATA in enumerate(DATA_BY_SEED):
        for data_name, data in DATA.items():
            # skip if already computed
            if (method_name, data_name, data_seed) in computed_keys:
                continue

            budgets = [int(b * len(data) * len(data[0]["scores"])) for b in BUDGETS_FRAC]
            parallelization_queue.append((method_name, method_fn, data_name, data_seed, data, budgets))



# do the computation
def _my_compute_method_with_data(x):
    method_name, method_fn, data_name, data_seed, data, budgets = x

    if isinstance(method_fn, functools.partial):
        n_seeds = method_fn.keywords.get("seeds") or 1
        assert isinstance(n_seeds, int), f"{n_seeds} is type {type(n_seeds)} but should be int"
    else:
        n_seeds = 1

    t0 = time.perf_counter()
    all_seed_results = []
    for s in range(n_seeds):
        data_cur = shuffle_items(data, seed=s) if n_seeds > 1 else data
        # meta-evaluate against the ground truth, which differs from the observed scores e.g. under annotator drift
        model_scores_all = {model: [item.get("scores_true", item["scores"])[model] for item in data_cur] for model in data_cur[0]["scores"]}
        model_scores_subset_at_budgets = method_fn(data_cur, budgets, seed=s)
        res_seed = [
            efficient_eval_bench.utils.meta_evaluate_scores_subset(model_scores_subset_at_budget, model_scores_all)
            for model_scores_subset_at_budget in model_scores_subset_at_budgets
        ]
        all_seed_results.append(res_seed)
    runtime = time.perf_counter() - t0
    runtime_avg = (runtime / n_seeds) / len(budgets) if budgets else (runtime / n_seeds)

    # Average meta-eval metrics across seeds for each budget
    results_local = []
    for b_idx in range(len(budgets)):
        b_dict = {}
        for key in all_seed_results[0][b_idx]:
            vals = [all_seed_results[s][b_idx][key] for s in range(n_seeds)]
            valid_vals = [v for v in vals if not np.isnan(v)]
            b_dict[key] = float(np.mean(valid_vals)) if valid_vals else float("nan")
        results_local.append(b_dict)

    return (method_name, results_local, data_name, data_seed, runtime_avg)

# start the large scenarios first so that they don't end up as the long tail
parallelization_queue.sort(key=lambda x: -len(x[4][0]["scores"]))

# raw per-seed results for 04-stat_tests.py, written as they come so that an interrupted run can be resumed
with open(RAW_PATH, "w") as f:
    for r in raw_rows:
        f.write(json.dumps(r) + "\n")

with multiprocessing.Pool(args.workers) as pool, open(RAW_PATH, "a") as f:
    for method_name, results_local, data_name, data_seed, runtime_avg in tqdm.tqdm(
        pool.imap_unordered(_my_compute_method_with_data, parallelization_queue),
        total=len(parallelization_queue),
    ):
        rows = [
            {
                "method": method_name,
                "scenario": data_name,
                "data_seed": data_seed,
                "budget_frac": round(float(budget_frac), 4),
                **b_dict,
                "time": runtime_avg,
            }
            for budget_frac, b_dict in zip(BUDGETS_FRAC, results_local)
        ]
        raw_rows.extend(rows)
        f.write("".join(json.dumps(r) + "\n" for r in rows))
        f.flush()

# collate the results across data seeds
grouped_results: dict[tuple[str, str], list[dict]] = {}
for r in raw_rows:
    if r["data_seed"] < args.data_seeds:
        grouped_results.setdefault((r["method"], r["scenario"]), []).append(r)

METRIC_KEYS = ["corr_tau", "corr_wtau", "mae_model", "mae_item"]
results_out: dict[str, dict[str, dict[str, float]]] = {}
for (method_name, data_name), rows in tqdm.tqdm(grouped_results.items()):
    entry = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        entry[key] = float(np.mean(vals)) if vals else float("nan")
    entry["time"] = statistics.mean(r["time"] for r in rows)
    results_out.setdefault(method_name, {})[data_name] = entry

with open("computed/01-results_all.toml", "w") as f:
    toml.dump(results_out, f)
