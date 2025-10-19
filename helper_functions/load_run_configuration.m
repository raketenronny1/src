function config = load_run_configuration(projectRoot, overrides)
%LOAD_RUN_CONFIGURATION Load project-wide configuration for analysis phases.
%
%   CONFIG = LOAD_RUN_CONFIGURATION(PROJECTROOT) reads the configuration
%   file located at PROJECTROOT/config/project_config.json (if present) and
%   merges it with internal defaults. The returned CONFIG struct provides
%   resolved values for commonly shared settings such as class labels,
%   metric name lists, and cross-validation fold counts.
%
%   CONFIG = LOAD_RUN_CONFIGURATION(PROJECTROOT, OVERRIDES) applies
%   selected override values on top of the loaded configuration. Supported
%   override fields include:
%       - configFile: path to an alternative JSON configuration file
%       - positiveClassLabel / negativeClassLabel
%       - phase2Metrics, phase2MetricsPreset, phase2OuterFolds,
%         phase2InnerFolds
%       - phase3Metrics, phase3MetricsPreset, phase3ProbeMetrics,
%         phase3ProbeMetricsPreset
%
%   The helper tolerates missing configuration files by falling back to
%   internal defaults and emitting a warning.

    if nargin < 1 || isempty(projectRoot)
        projectRoot = pwd;
    end
    if nargin < 2
        overrides = struct();
    end

    config = get_default_configuration();

    configFile = fullfile(projectRoot, 'config', 'project_config.json');
    if isfield(overrides, 'configFile') && ~isempty(overrides.configFile)
        configFile = overrides.configFile;
    end

    if isfile(configFile)
        try
            jsonText = fileread(configFile);
            fileConfig = jsondecode(jsonText);
            config = merge_structs_recursive(config, fileConfig);
        catch ME
            warning('load_run_configuration:FailedToParse', ...
                'Failed to parse configuration file %s (%s). Using defaults.', ...
                configFile, ME.message);
        end
    else
        warning('load_run_configuration:MissingFile', ...
            'Configuration file %s not found. Using defaults.', configFile);
    end

    % Apply overrides supplied directly via function input
    if isfield(overrides, 'positiveClassLabel') && ~isempty(overrides.positiveClassLabel)
        config.classLabels.positive = overrides.positiveClassLabel;
    end
    if isfield(overrides, 'negativeClassLabel') && ~isempty(overrides.negativeClassLabel)
        config.classLabels.negative = overrides.negativeClassLabel;
    end

    % Phase 2 overrides
    if isfield(overrides, 'phase2OuterFolds') && ~isempty(overrides.phase2OuterFolds)
        config.phase2.outerFolds = overrides.phase2OuterFolds;
    end
    if isfield(overrides, 'phase2InnerFolds') && ~isempty(overrides.phase2InnerFolds)
        config.phase2.innerFolds = overrides.phase2InnerFolds;
    end
    if isfield(overrides, 'phase2Metrics') && ~isempty(overrides.phase2Metrics)
        config.phase2.metrics = overrides.phase2Metrics;
    elseif isfield(overrides, 'phase2MetricsPreset') && ~isempty(overrides.phase2MetricsPreset)
        config.phase2.metricsPreset = overrides.phase2MetricsPreset;
    end

    % Phase 3 overrides
    if isfield(overrides, 'phase3Metrics') && ~isempty(overrides.phase3Metrics)
        config.phase3.metrics = overrides.phase3Metrics;
    elseif isfield(overrides, 'phase3MetricsPreset') && ~isempty(overrides.phase3MetricsPreset)
        config.phase3.metricsPreset = overrides.phase3MetricsPreset;
    end
    if isfield(overrides, 'phase3ProbeMetrics') && ~isempty(overrides.phase3ProbeMetrics)
        config.phase3.probeMetrics = overrides.phase3ProbeMetrics;
    elseif isfield(overrides, 'phase3ProbeMetricsPreset') && ~isempty(overrides.phase3ProbeMetricsPreset)
        config.phase3.probeMetricsPreset = overrides.phase3ProbeMetricsPreset;
    end

    % Resolve metric presets after all overrides applied
    config = resolve_metric_presets(config);
end

function config = resolve_metric_presets(config)
    presets = config.metricPresets;

    if ~isfield(config.phase2, 'metrics') || isempty(config.phase2.metrics)
        config.phase2.metrics = fetch_metrics_from_preset(presets, config.phase2.metricsPreset, presets.default);
    else
        config.phase2.metrics = ensure_cellstr(config.phase2.metrics);
    end
    if ~isfield(config.phase3, 'metrics') || isempty(config.phase3.metrics)
        config.phase3.metrics = fetch_metrics_from_preset(presets, config.phase3.metricsPreset, presets.default);
    else
        config.phase3.metrics = ensure_cellstr(config.phase3.metrics);
    end
    if ~isfield(config.phase3, 'probeMetrics') || isempty(config.phase3.probeMetrics)
        fallback = fetch_metrics_from_preset(presets, config.phase3.metricsPreset, presets.default);
        config.phase3.probeMetrics = fetch_metrics_from_preset(presets, config.phase3.probeMetricsPreset, fallback);
    else
        config.phase3.probeMetrics = ensure_cellstr(config.phase3.probeMetrics);
    end
end

function metrics = fetch_metrics_from_preset(presets, presetName, fallback)
    metrics = fallback;
    if nargin < 3 || isempty(fallback)
        fallback = {};
    end
    if isfield(presets, presetName)
        metrics = presets.(presetName);
    elseif ~isempty(fallback)
        metrics = fallback;
    end
    metrics = ensure_cellstr(metrics);
end

function config = get_default_configuration()
    config = struct();
    config.classLabels = struct('positive', 3, 'negative', 1);

    defaultMetrics = {'Accuracy','Sensitivity_WHO3','Specificity_WHO1', ...
        'PPV_WHO3','NPV_WHO1','F1_WHO3','F2_WHO3','AUC'};

    config.metricPresets = struct();
    config.metricPresets.default = defaultMetrics;
    config.metricPresets.phase2_model_selection = defaultMetrics;
    config.metricPresets.phase3_final_evaluation = defaultMetrics;
    config.metricPresets.probe_level_summary = defaultMetrics;

    config.phase2 = struct();
    config.phase2.outerFolds = 5;
    config.phase2.innerFolds = 3;
    config.phase2.metricsPreset = 'phase2_model_selection';
    config.phase2.metrics = {};

    config.phase3 = struct();
    config.phase3.metricsPreset = 'phase3_final_evaluation';
    config.phase3.metrics = {};
    config.phase3.probeMetricsPreset = 'probe_level_summary';
    config.phase3.probeMetrics = {};
end

function out = merge_structs_recursive(base, override)
    out = base;
    if ~isstruct(override)
        out = override;
        return;
    end
    fields = fieldnames(override);
    for i = 1:numel(fields)
        name = fields{i};
        value = override.(name);
        if isfield(base, name) && isstruct(base.(name)) && isstruct(value)
            out.(name) = merge_structs_recursive(base.(name), value);
        else
            out.(name) = value;
        end
    end
end

function c = ensure_cellstr(val)
    if isstring(val)
        c = cellstr(val);
    elseif ischar(val)
        c = {val};
    elseif iscell(val)
        c = val;
    elseif isnumeric(val)
        c = num2cell(val);
    else
        c = {};
    end
end
