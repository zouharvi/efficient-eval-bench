(async () => {
  "use strict";
  const resultsSection = document.getElementById("results");
  const controls = document.querySelectorAll(".controls input, .controls select, [data-sort]");
  resultsSection.setAttribute("aria-busy", "true");
  controls.forEach(control => { control.disabled = true; });
  let data;
  try {
    const response = await fetch("../computed/01-results_all.toml", { cache: "no-store" });
    if (!response.ok) throw new Error(`Results request failed (HTTP ${response.status}).`);
    const results = parseResultsToml(await response.text());
    const available = new Set(Object.values(results).flatMap(entries => Object.keys(entries)));
    const scenarios = window.BENCHMARK_SCENARIOS.filter(item => available.has(item.key));
    const known = new Set(scenarios.map(item => item.key));
    for (const key of [...available].sort()) {
      if (!known.has(key)) scenarios.push({ key, label: key, group: "Other scenarios", description: "See the repository for this scenario’s definition." });
    }
    data = { results, scenarios };
  } catch (error) {
    document.getElementById("result-status").textContent = "Results could not be loaded.";
    const message = document.getElementById("load-error");
    message.textContent = window.location.protocol === "file:"
      ? "To load the results, serve the repository root with python3 -m http.server 8000, then open http://localhost:8000/website/."
      : `${error.message} Make sure computed/01-results_all.toml is served alongside website/, then reload this page.`;
    message.hidden = false;
    return;
  } finally {
    resultsSection.setAttribute("aria-busy", "false");
  }

  const baseline = "Random (uniform)";
  const metrics = ["mae_item", "mae_model", "corr_tau", "corr_wtau"];
  const metricNames = {
    method: "method name", corr_tau: "Global Model Ranking", corr_wtau: "Top-Weighted Model Ranking",
    mae_model: "Benchmark-level Score Estimation", mae_item: "Item-level Score Estimation",
  };
  const lowerIsBetter = key => key.startsWith("mae_");
  const scenario = document.getElementById("scenario");
  const search = document.getElementById("search");
  const compare = document.getElementById("compare");
  const showAll = document.getElementById("show-all");
  const status = document.getElementById("result-status");
  const groups = new Map();
  scenario.replaceChildren();
  for (const item of data.scenarios) {
    if (!groups.has(item.group)) {
      const group = document.createElement("optgroup");
      group.label = item.group;
      groups.set(item.group, group);
      scenario.append(group);
    }
    groups.get(item.group).append(new Option(item.label, item.key));
  }
  for (const id of ["method-count", "statement-method-count"]) document.getElementById(id).textContent = Object.keys(data.results).length;
  for (const id of ["scenario-count", "statement-scenario-count"]) document.getElementById(id).textContent = data.scenarios.length;
  controls.forEach(control => { control.disabled = false; });
  let sortKey = "corr_tau";
  let ascending = false;
  let expanded = false;

  const requestedScenario = new URLSearchParams(window.location.search).get("scenario");
  if (data.scenarios.some(item => item.key === requestedScenario)) scenario.value = requestedScenario;

  function numericValue(method, key) {
    return data.results[method]?.[scenario.value]?.[key] ?? null;
  }

  function sortedMethods() {
    return Object.keys(data.results).sort((a, b) => {
      if (sortKey === "method") return (ascending ? 1 : -1) * a.localeCompare(b);
      const av = numericValue(a, sortKey);
      const bv = numericValue(b, sortKey);
      if (!Number.isFinite(av)) return Number.isFinite(bv) ? 1 : a.localeCompare(b);
      if (!Number.isFinite(bv)) return -1;
      return (ascending ? av - bv : bv - av) || a.localeCompare(b);
    });
  }

  function format(value) {
    if (!Number.isFinite(value)) return "—";
    const rounded = Math.abs(value) < 0.0005 ? 0 : value;
    return rounded.toFixed(3).replace("-", "−");
  }

  function makeRow(method, rank) {
    const row = document.createElement("tr");
    if (method === baseline) row.className = "baseline-row";
    const number = document.createElement("td");
    number.className = "rank";
    number.textContent = sortKey === "method" ? "—" : String(rank);
    row.append(number);
    const name = document.createElement("th");
    name.scope = "row";
    name.textContent = method;
    if (method === baseline) {
      const label = document.createElement("span");
      label.className = "baseline-label";
      label.textContent = "reference";
      name.append(label);
    }
    row.append(name);
    for (const key of metrics) {
      const cell = document.createElement("td");
      let value = numericValue(method, key);
      const rawValue = value;
      if (key === sortKey) cell.classList.add("sorted-cell");
      if (compare.checked) {
        const ref = numericValue(baseline, key);
        value = Number.isFinite(value) && Number.isFinite(ref)
          ? (lowerIsBetter(key) ? ref - value : value - ref) : null;
        if (value >= 0.0005) cell.classList.add("gain");
        if (value <= -0.0005) cell.classList.add("loss");
      }
      cell.textContent = `${compare.checked && value >= 0.0005 ? "+" : ""}${format(value)}`;
      cell.title = `${metricNames[key]}: ${Number.isFinite(rawValue) ? rawValue : "not available"}`;
      row.append(cell);
    }
    return row;
  }

  function render() {
    const current = data.scenarios.find(item => item.key === scenario.value);
    const displayedAscending = compare.checked && lowerIsBetter(sortKey) ? !ascending : ascending;
    document.getElementById("scenario-description").textContent = current.description;
    document.getElementById("table-caption").textContent = `${current.label}. ${compare.checked ? "Improvements relative to uniform random sampling; positive values are better in every column." : "Absolute scores."} Sorted by ${metricNames[sortKey]}, ${displayedAscending ? "ascending" : "descending"}.`;
    const all = sortedMethods();
    // Competition ranks use unrounded scores; filtering never changes a method's rank.
    const ranks = new Map();
    let previousValue;
    let rank = 1;
    all.forEach((method, index) => {
      const value = numericValue(method, sortKey);
      if (index === 0 || value !== previousValue) rank = index + 1;
      ranks.set(method, rank);
      previousValue = value;
    });
    const query = search.value.trim().toLocaleLowerCase();
    const matched = all.filter(method => method !== baseline && method.toLocaleLowerCase().includes(query));
    const shown = expanded || query ? matched : matched.slice(0, 15);
    document.getElementById("baseline-rows").replaceChildren(makeRow(baseline, ranks.get(baseline)));
    document.getElementById("result-rows").replaceChildren(...shown.map(method => makeRow(method, ranks.get(method))));
    document.getElementById("empty-state").hidden = matched.length > 0;
    status.textContent = `${shown.length} of ${matched.length} methods + random baseline · sorted by ${metricNames[sortKey]}`;
    showAll.hidden = Boolean(query) || matched.length <= 15;
    showAll.textContent = expanded ? "Show fewer methods" : `Show all ${matched.length} methods`;
    showAll.setAttribute("aria-expanded", String(expanded));
    document.getElementById("reading-note").textContent = compare.checked
      ? "Positive values mean improvement over random sampling in every column. For MAE, the value is the reduction in error. Sorting uses full precision."
      : "Uniform random sampling is the reference baseline. Values are rounded to three decimals; sorting uses full precision.";
    for (const button of document.querySelectorAll("[data-sort]")) {
      const key = button.dataset.sort;
      const direction = key === sortKey ? (displayedAscending ? "ascending" : "descending") : "none";
      button.parentElement.setAttribute("aria-sort", direction);
      button.setAttribute("aria-label", `Sort by ${metricNames[key]}`);
    }
  }

  scenario.addEventListener("change", () => {
    render();
    const url = new URL(window.location.href);
    if (scenario.value === "base") url.searchParams.delete("scenario");
    else url.searchParams.set("scenario", scenario.value);
    try { window.history.replaceState(null, "", url); } catch { /* File previews may restrict history. */ }
  });
  search.addEventListener("input", render);
  compare.addEventListener("change", render);
  showAll.addEventListener("click", () => { expanded = !expanded; render(); });
  document.querySelectorAll("[data-sort]").forEach(button => {
    button.addEventListener("click", () => {
      const key = button.dataset.sort;
      ascending = key === sortKey ? !ascending : key === "method" || lowerIsBetter(key);
      sortKey = key;
      render();
    });
  });
  render();
})();
