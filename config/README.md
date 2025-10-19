# Project configuration

The JSON file `project_config.json` defines cross-cutting defaults consumed by
`run_phase2_model_selection.m`, `run_phase3_final_evaluation.m`, and their
supporting helpers. The `load_run_configuration` utility reads this file and
merges it with built-in defaults. Override values can also be provided when
calling the phase scripts by setting fields on the `cfg` struct (for example,
`cfg.phase2OuterFolds = 7`).

## Top-level sections

- `classLabels`
  - `positive`: Label treated as the positive (e.g. WHO‑3) class when computing
    metrics.
  - `negative`: Label treated as the negative (e.g. WHO‑1) class. Used for
    probe-level predictions and as a fallback when scores are missing.
- `metricPresets`: Named collections of metric identifiers that the helper
  functions can reuse. All presets listed here become available through
  `metric_name_presets()`.
- `phase2`
  - `outerFolds`: Number of folds used in the outer cross-validation loop.
  - `innerFolds`: Number of folds used in the inner tuning loop.
  - `metricsPreset`: Name of the metric preset applied during Phase 2 scoring.
- `phase3`
  - `metricsPreset`: Preset to evaluate on the test set.
  - `probeMetricsPreset`: Preset used when summarising probe-level predictions.

Any of the presets can be replaced with a `metrics` or `probeMetrics` array in
the JSON file if you prefer to spell out the metric list explicitly.
