/* Parser for the runner's numeric TOML tables, not a general-purpose TOML parser.
   Unsupported syntax fails explicitly rather than producing partial results. */
(() => {
  "use strict";
  function parseResultsToml(text) {
    const results = Object.create(null);
    const key = '(?:"(?:[^"\\\\]|\\\\.)*"|[A-Za-z0-9_-]+)';
    const header = new RegExp(`^\\[\\s*(${key})\\s*\\.\\s*(${key})\\s*\\]\\s*(?:#.*)?$`);
    const number = /^[+-]?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|inf|nan)$/;
    const decode = value => value.startsWith('"') ? JSON.parse(value) : value;
    let current;
    for (const [index, raw] of text.replace(/^\uFEFF/, "").split(/\r?\n/).entries()) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      const fail = () => { throw new Error(`Unsupported or invalid results TOML at line ${index + 1}`); };
      const table = header.exec(line);
      if (table) {
        const method = decode(table[1]);
        const scenario = decode(table[2]);
        if (!results[method]) results[method] = Object.create(null);
        if (Object.hasOwn(results[method], scenario)) fail();
        current = results[method][scenario] = Object.create(null);
        continue;
      }
      const assignment = /^([A-Za-z0-9_-]+)\s*=\s*([^#]+?)(?:\s*#.*)?$/.exec(line);
      if (!assignment || !current) fail();
      const name = assignment[1];
      const value = assignment[2].trim().replace(/(?<=\d)_(?=\d)/g, "");
      if (!number.test(value) || Object.hasOwn(current, name)) fail();
      const numeric = Number(value);
      current[name] = Number.isFinite(numeric) ? numeric : null;
    }
    if (!results["Random (uniform)"]?.base) throw new Error("Results are missing the uniform baseline or base scenario.");
    for (const scenarios of Object.values(results)) {
      for (const values of Object.values(scenarios)) {
        for (const metric of ["corr_tau", "corr_wtau", "mae_model", "mae_item"]) {
          if (!Object.hasOwn(values, metric)) throw new Error(`Results are missing ${metric}.`);
        }
      }
    }
    return results;
  }
  globalThis.parseResultsToml = parseResultsToml;
})();
