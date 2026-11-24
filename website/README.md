# Efficient Eval Bench website

A single page for exploring the benchmark results.

To preview it, run this from the repository root:

```sh
python3 -m http.server 8000
```

Open http://localhost:8000/website/.

The page loads `computed/01-results_all.toml` directly. After updating the results,
reload the page to see the changes.

## Editing

- `index.html`: page text and layout
- `style.css`: colours, fonts, and spacing
- `app.js`: loading, filtering, sorting, and comparisons
- `scenarios.js`: scenario names and descriptions
- `results-parser.js`: reads the results TOML

Keep scenario descriptions in sync with `efficient_eval_bench/data/__init__.py`.

## Hosting

Serve `website/` alongside `computed/01-results_all.toml`, keeping the same
directory structure as the repository.
