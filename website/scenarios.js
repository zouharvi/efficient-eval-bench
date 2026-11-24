// Descriptive metadata only; numerical results are fetched from computed/.
window.BENCHMARK_SCENARIOS = [
  {
    "key": "base",
    "label": "Base scenario",
    "group": "Reference",
    "description": "16 models, 500 items, continuous scores. Automated metric noise σ = 0.05."
  },
  {
    "key": "4 models",
    "label": "4 models",
    "group": "Model count",
    "description": "4 models and 500 items. All other parameters match the base configuration."
  },
  {
    "key": "16 models",
    "label": "16 models",
    "group": "Model count",
    "description": "16 models and 500 items. All other parameters match the base configuration."
  },
  {
    "key": "64 models",
    "label": "64 models",
    "group": "Model count",
    "description": "64 models and 500 items. All other parameters match the base configuration."
  },
  {
    "key": "256 models",
    "label": "256 models",
    "group": "Model count",
    "description": "256 models and 500 items. All other parameters match the base configuration."
  },
  {
    "key": "Item heteroscedasticity 1",
    "label": "Item-dependent noise · γ = 1",
    "group": "Heteroscedasticity",
    "description": "Harder items receive more noise. Heteroscedasticity coefficient γ = 1; 16 models, 500 items."
  },
  {
    "key": "Item heteroscedasticity 2",
    "label": "Item-dependent noise · γ = 2",
    "group": "Heteroscedasticity",
    "description": "Harder items receive more noise. Heteroscedasticity coefficient γ = 2; 16 models, 500 items."
  },
  {
    "key": "Item heteroscedasticity 3",
    "label": "Item-dependent noise · γ = 4",
    "group": "Heteroscedasticity",
    "description": "Harder items receive more noise. Heteroscedasticity coefficient γ = 4; 16 models, 500 items."
  },
  {
    "key": "Model heteroscedasticity 1",
    "label": "Model-dependent noise · γ = 1",
    "group": "Heteroscedasticity",
    "description": "Models with higher latent ability receive more noise. Heteroscedasticity coefficient γ = 1; 16 models, 500 items."
  },
  {
    "key": "Model heteroscedasticity 2",
    "label": "Model-dependent noise · γ = 2",
    "group": "Heteroscedasticity",
    "description": "Models with higher latent ability receive more noise. Heteroscedasticity coefficient γ = 2; 16 models, 500 items."
  },
  {
    "key": "Model heteroscedasticity 3",
    "label": "Model-dependent noise · γ = 4",
    "group": "Heteroscedasticity",
    "description": "Models with higher latent ability receive more noise. Heteroscedasticity coefficient γ = 4; 16 models, 500 items."
  },
  {
    "key": "Metric noise 0.5",
    "label": "Metric noise · σ = 0.5",
    "group": "Automated metric",
    "description": "Automated metric noise σ = 0.5, compared with 0.05 in the base scenario. Scores are clipped to [0, 1]."
  },
  {
    "key": "Metric noise 1.0",
    "label": "Metric noise · σ = 1.0",
    "group": "Automated metric",
    "description": "Automated metric noise σ = 1.0, compared with 0.05 in the base scenario. Scores are clipped to [0, 1]."
  },
  {
    "key": "Metric noise 2.0",
    "label": "Metric noise · σ = 2.0",
    "group": "Automated metric",
    "description": "Automated metric noise σ = 2.0, compared with 0.05 in the base scenario. Scores are clipped to [0, 1]."
  },
  {
    "key": "Metric by group",
    "label": "Metric quality varies by group",
    "group": "Automated metric",
    "description": "Items are randomly assigned to two groups with metric noise σ = 0.02 or 0.40."
  },
  {
    "key": "Annotator strict 1",
    "label": "Annotator strictness · 0.25",
    "group": "Annotator behaviour",
    "description": "The downward score shift grows from 0 to 0.25 across item order. Evaluation uses the clean scores."
  },
  {
    "key": "Annotator strict 2",
    "label": "Annotator strictness · 0.5",
    "group": "Annotator behaviour",
    "description": "The downward score shift grows from 0 to 0.5 across item order. Evaluation uses the clean scores."
  },
  {
    "key": "Annotator strict 3",
    "label": "Annotator strictness · 1",
    "group": "Annotator behaviour",
    "description": "The downward score shift grows from 0 to 1 across item order. Evaluation uses the clean scores."
  },
  {
    "key": "Annotator inconsistency 1",
    "label": "Annotator inconsistency · 0.5",
    "group": "Annotator behaviour",
    "description": "Annotation noise standard deviation increases by 0.5 across item order, from a starting value of 0.05."
  },
  {
    "key": "Annotator inconsistency 2",
    "label": "Annotator inconsistency · 1",
    "group": "Annotator behaviour",
    "description": "Annotation noise standard deviation increases by 1 across item order, from a starting value of 0.05."
  },
  {
    "key": "Annotator inconsistency 3",
    "label": "Annotator inconsistency · 2",
    "group": "Annotator behaviour",
    "description": "Annotation noise standard deviation increases by 2 across item order, from a starting value of 0.05."
  },
  {
    "key": "Annotator replacement 1",
    "label": "Annotator replacement · 5%",
    "group": "Annotator behaviour",
    "description": "At each item, an annotator is replaced with probability 5%. Annotators vary in bias and noise."
  },
  {
    "key": "Annotator replacement 2",
    "label": "Annotator replacement · 10%",
    "group": "Annotator behaviour",
    "description": "At each item, an annotator is replaced with probability 10%. Annotators vary in bias and noise."
  },
  {
    "key": "Annotator replacement 3",
    "label": "Annotator replacement · 20%",
    "group": "Annotator behaviour",
    "description": "At each item, an annotator is replaced with probability 20%. Annotators vary in bias and noise."
  },
  {
    "key": "Binary",
    "label": "Binary scores",
    "group": "Score scale",
    "description": "Scores are rounded to 0 or 1. 16 models and 500 items."
  },
  {
    "key": "Likert",
    "label": "Likert scores",
    "group": "Score scale",
    "description": "Scores are rounded to five values: 0, 0.25, 0.5, 0.75, and 1. 16 models and 500 items."
  }
];
