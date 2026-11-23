import functools
from collections.abc import Callable
from dataclasses import dataclass

from efficient_eval_bench.methods import Data


@dataclass(frozen=True)
class AnnotatorDrift:
    """Simulates human annotation that changes over (annotation) time.

    Items are annotated in dataset order, so item index i corresponds to time t = i / (items - 1) in [0, 1].
    Each item is annotated (for all models) by one annotator drawn from the current pool.
    The observed score is clip(true + annotator_bias - strictness_drift * t + N(0, noise_sd(t)), 0, 1),
    where noise_sd(t) = annotator_noise + inconsistency_drift * t.

    The clean (pre-annotation) score is kept as ground truth in `scores_true`.
    """

    n_annotators: int = 5
    # sd of the per-annotator constant bias (drawn when the annotator joins the pool)
    bias_scale: float = 0.0
    # per-annotation noise sd at t=0 (per-annotator, drawn uniformly from this range)
    noise_range: tuple[float, float] = (0.05, 0.05)
    # total downward shift of observed scores from first to last item (annotators get stricter)
    strictness_drift: float = 0.0
    # total increase in noise sd from first to last item (annotators get less consistent)
    inconsistency_drift: float = 0.0
    # probability, per item, that a random annotator in the pool is replaced by a fresh one
    replacement_rate: float = 0.0


def load_data_synth(
    seed=0,
    models=16,
    items=500,
    model_heterosced: float = 0.0,
    item_heterosced: float = 0.0,
    bins=None,
    embedding_dim=64,
    metric_noise: float = 0.05,
    metric_group_noise: tuple[float, ...] | None = None,
    annotator_drift: AnnotatorDrift | None = None,
) -> Data:
    """
    Synthetic scenario generator. With default arguments the output is identical to the original generator.

    - model_heterosced: model-dependent noise. Noise sd is multiplied by exp(gamma * z) where z is the
      standardized model latent ability, normalized so that the average multiplier is 1. 0 turns it off;
      negative values reverse the direction.
    - item_heterosced: item-dependent noise. Noise sd is multiplied by exp(gamma * difficulty) where
      difficulty = -item_latent, normalized so that the average multiplier is 1. 0 turns it off.
    - metric_noise: sd of the automated metric's noise around the true score (metric reliability).
    - metric_group_noise: distribution-dependent metric reliability. Items are split uniformly at random into
      len(metric_group_noise) groups and the metric noise sd for group k is metric_group_noise[k].
      The group is exposed via the `domain` field ("synth/g{k}"). Overrides metric_noise.
    - annotator_drift: corrupts the observed human scores with time-varying annotator behaviour,
      see AnnotatorDrift. Ground truth is stored in `scores_true`.
    """
    import numpy as np

    random = np.random.RandomState(seed)
    # separate streams so that the new options do not change the base data
    random_groups = np.random.RandomState(seed + 1_000)
    random_annot = np.random.RandomState(seed + 2_000)
    if bins is not None:
        bins = np.asarray(bins, dtype=float)

    def _bin(x):
        # get closest bin, not digitize
        return bins[np.argmin(np.abs(bins - x))] if bins is not None else x

    models_latent = np.clip(random.normal(loc=0.70, scale=0.25, size=models), 0, 1)
    items_latent = random.normal(loc=0, scale=1, size=items)

    model_latent_mean = np.mean(models_latent)
    model_latent_std = np.std(models_latent)
    models_z = (models_latent - model_latent_mean) / model_latent_std if model_latent_std > 0 else np.zeros_like(models_latent)
    models_noise_mult = np.exp(model_heterosced * models_z)
    models_noise_mult /= np.mean(models_noise_mult)

    # Fixed random projection matrix for generating synthetic item embeddings
    W = random.normal(size=(models, embedding_dim))

    # item-dependent noise multiplier, harder items (lower latent) are noisier
    # E[exp(g*Z)] = exp(g^2/2) for Z~N(0,1), so normalize to keep the average noise level comparable
    items_noise_mult = np.exp(-item_heterosced * items_latent - item_heterosced**2 / 2)

    if metric_group_noise is not None:
        items_group = random_groups.randint(0, len(metric_group_noise), size=items)
    else:
        items_group = None

    if annotator_drift is not None:
        def _new_annotator():
            return (random_annot.normal(0, annotator_drift.bias_scale), random_annot.uniform(*annotator_drift.noise_range))

        annotators = [_new_annotator() for _ in range(annotator_drift.n_annotators)]

    data_out: Data = []
    for item_i, item_latent in enumerate(items_latent):
        scores_dict = {}
        scores_true_dict = {}
        scores_metric_dict = {}

        if annotator_drift is not None:
            assert annotators # type: ignore
            assert _new_annotator # type: ignore

            t = item_i / max(1, items - 1)
            if random_annot.rand() < annotator_drift.replacement_rate:
                annotators[random_annot.randint(len(annotators))] = _new_annotator()
            annot_bias, annot_noise = annotators[random_annot.randint(len(annotators))]
            annot_shift = annot_bias - annotator_drift.strictness_drift * t
            annot_noise = annot_noise + annotator_drift.inconsistency_drift * t

        cur_metric_noise = metric_group_noise[items_group[item_i]] if items_group is not None else metric_noise # type: ignore

        for model_i, model_latent in enumerate(models_latent):
            error = random.normal(loc=0, scale=model_latent_mean * models_noise_mult[model_i] * items_noise_mult[item_i])
            score_raw = np.clip(model_latent + item_latent + error, 0, 1)
            score = _bin(score_raw)
            model_name = f"model_{model_i + 1}"
            scores_true_dict[model_name] = float(score)
            scores_metric_dict[model_name] = {"metric": float(np.clip(score + random.normal(loc=0, scale=cur_metric_noise), 0, 1))}

            if annotator_drift is not None:
                # annotators perceive the continuous quality, then report on the same scale (bins)
                score_obs = _bin(np.clip(score_raw + annot_shift + random_annot.normal(0, annot_noise), 0, 1)) # type: ignore
                scores_dict[model_name] = float(score_obs)
            else:
                scores_dict[model_name] = float(score)

        score_vec = np.array([scores_true_dict[f"model_{model_i + 1}"] for model_i in range(models)], dtype=float)
        emb = score_vec @ W + random.normal(scale=0.05, size=embedding_dim)
        emb_norm = float(np.linalg.norm(emb))
        emb = (emb / (emb_norm if emb_norm > 0 else 1.0)).tolist()

        item_out = {
            "scores": scores_dict,
            "scores_metrics": scores_metric_dict,
            "cost": 1,
            "embedding": emb,
            "domain": f"synth/g{items_group[item_i]}" if items_group is not None else "synth",
        }
        if annotator_drift is not None:
            item_out["scores_true"] = scores_true_dict
        data_out.append(item_out)  # type: ignore
    return data_out


def load_data_subset2evaluate_translation():
    import subset2evaluate.utils

    data = subset2evaluate.utils.load_data_wmt("wmt25", "en-cs_CZ")
    data_new = []
    for line in data:
        data_new.append(
            {
                "src": line["src"],
                "tgt": line["tgt"],
                "scores": {model: line["scores"][model]["human"] / 100 for model in line["scores"]},
                "scores_metrics": {model: {"metric": line["scores"][model]["MetricX-25"]} for model in line["scores"]},
                "cost": 1,
                "domain": "wmt25/en-cs",
            }
        )
    return data_new


def load_data_subset2evaluate_summeval():
    import subset2evaluate.utils

    data = subset2evaluate.utils.load_data_summeval(normalize=False, load_extra=False)
    data_new = []
    for line in data:
        data_new.append(
            {
                "src": line["src"],
                "tgt": line["tgt"],
                "scores": {model: line["scores"][model]["human_overall"] for model in line["scores"]},
                "scores_metrics": {model: {"metric": line["scores"][model]["unieval_overall"]} for model in line["scores"]},
                "cost": 1,
                "domain": "summeval",
            }
        )
    return data_new


load_data_base: Callable[..., Data] = functools.partial(load_data_synth)
load_data_binary: Callable[..., Data] = functools.partial(load_data_synth, bins=[0, 1])
load_data_likert: Callable[..., Data] = functools.partial(load_data_synth, bins=[0, 0.25, 0.5, 0.75, 1])
load_data_4models: Callable[..., Data] = functools.partial(load_data_synth, models=4, model_heterosced=0.0)
load_data_16models: Callable[..., Data] = functools.partial(load_data_synth, models=16, model_heterosced=0.0)
load_data_64models: Callable[..., Data] = functools.partial(load_data_synth, models=64, model_heterosced=0.0)
load_data_256models: Callable[..., Data] = functools.partial(load_data_synth, models=256, model_heterosced=0.0)

# item-dependent heteroscedasticity: harder items are noisier
load_data_item_hetero1: Callable[..., Data] = functools.partial(load_data_synth, item_heterosced=1.0)
load_data_item_hetero2: Callable[..., Data] = functools.partial(load_data_synth, item_heterosced=2.0)
load_data_item_hetero3: Callable[..., Data] = functools.partial(load_data_synth, item_heterosced=4.0)

# model-dependent heteroscedasticity: better models are noisier
load_data_model_hetero1: Callable[..., Data] = functools.partial(load_data_synth, model_heterosced=1.0)
load_data_model_hetero2: Callable[..., Data] = functools.partial(load_data_synth, model_heterosced=2.0)
load_data_model_hetero3: Callable[..., Data] = functools.partial(load_data_synth, model_heterosced=4.0)

# automated metric reliability: increasingly noisy metric (default is 0.05)
load_data_metric_noise1: Callable[..., Data] = functools.partial(load_data_synth, metric_noise=0.5)
load_data_metric_noise2: Callable[..., Data] = functools.partial(load_data_synth, metric_noise=1.0)
load_data_metric_noise3: Callable[..., Data] = functools.partial(load_data_synth, metric_noise=2.0)

# distribution-dependent metric reliability: metric is good on half of the items and poor on the other half
load_data_metric_groups: Callable[..., Data] = functools.partial(load_data_synth, metric_group_noise=(0.02, 0.40))

# annotator drift
load_data_annot_strict1: Callable[..., Data] = functools.partial(
    load_data_synth, annotator_drift=AnnotatorDrift(strictness_drift=0.25)
)
load_data_annot_strict2: Callable[..., Data] = functools.partial(
    load_data_synth, annotator_drift=AnnotatorDrift(strictness_drift=0.5)
)
load_data_annot_strict3: Callable[..., Data] = functools.partial(
    load_data_synth, annotator_drift=AnnotatorDrift(strictness_drift=1.0)
)       
load_data_annot_inconsistency1: Callable[..., Data] = functools.partial(
    load_data_synth, annotator_drift=AnnotatorDrift(inconsistency_drift=0.5)
)
load_data_annot_inconsistency2: Callable[..., Data] = functools.partial(
    load_data_synth, annotator_drift=AnnotatorDrift(inconsistency_drift=1.0)
)
load_data_annot_inconsistency3: Callable[..., Data] = functools.partial(
    load_data_synth, annotator_drift=AnnotatorDrift(inconsistency_drift=2.0)
)
load_data_annot_replacement1: Callable[..., Data] = functools.partial(
    load_data_synth, annotator_drift=AnnotatorDrift(bias_scale=0.15, noise_range=(0.02, 0.20), replacement_rate=0.05)
)
load_data_annot_replacement2: Callable[..., Data] = functools.partial(
    load_data_synth, annotator_drift=AnnotatorDrift(bias_scale=0.15, noise_range=(0.02, 0.20), replacement_rate=0.10)
)
load_data_annot_replacement3: Callable[..., Data] = functools.partial(
    load_data_synth, annotator_drift=AnnotatorDrift(bias_scale=0.15, noise_range=(0.02, 0.20), replacement_rate=0.20)
)


_DATA: dict[str, Callable[..., Data]] = {
    "base": load_data_base,
    "4 models": load_data_4models,
    "16 models": load_data_16models,
    "64 models": load_data_64models,
    "256 models": load_data_256models,
    "Model heteroscedasticity 1": load_data_model_hetero1,
    "Model heteroscedasticity 2": load_data_model_hetero2,
    "Model heteroscedasticity 3": load_data_model_hetero3,
    "Item heteroscedasticity 1": load_data_item_hetero1,
    "Item heteroscedasticity 2": load_data_item_hetero2,
    "Item heteroscedasticity 3": load_data_item_hetero3,
    "Metric noise 0.5": load_data_metric_noise1,
    "Metric noise 1.0": load_data_metric_noise2,
    "Metric noise 2.0": load_data_metric_noise3,
    "Annotator strict 1": load_data_annot_strict1,
    "Annotator strict 2": load_data_annot_strict2,
    "Annotator strict 3": load_data_annot_strict3,
    "Annotator inconsistency 1": load_data_annot_inconsistency1,
    "Annotator inconsistency 2": load_data_annot_inconsistency2,
    "Annotator inconsistency 3": load_data_annot_inconsistency3,
    "Annotator replacement 1": load_data_annot_replacement1,
    "Annotator replacement 2": load_data_annot_replacement2,
    "Annotator replacement 3": load_data_annot_replacement3,
    # re-think?
    "Binary": load_data_binary,
    "Likert": load_data_likert,
    "Metric by group": load_data_metric_groups,
}


def get_data_dict_all(seed: int = 0) -> dict[str, Data]:
    return {k: v(seed=seed) for k, v in _DATA.items()}


def shuffle_items(data: Data, seed: int) -> Data:
    import numpy as np

    rng = np.random.default_rng(seed)
    return [data[i] for i in rng.permutation(len(data))]
