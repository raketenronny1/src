%%
% calculate_performance_metrics.m
%
% Helper function to calculate various classification performance metrics.
%
% INPUTS:
%   y_true                  - (N_samples x 1) True class labels.
%   y_pred                  - (N_samples x 1) Predicted class labels.
%   y_scores_positive_class - (N_samples x 1) Scores for the positive class
%                             (e.g., probabilities for WHO-3).
%   optionsOrPositive       - Either a struct containing configuration
%                             fields (see below) or a scalar specifying the
%                             positive class label for backwards
%                             compatibility.
%   metricNames             - Optional cell array with metric names. When
%                             omitted the function attempts to infer the
%                             list from OPTIONSORPOSITIVE.metricNames or a
%                             preset specified via
%                             OPTIONSORPOSITIVE.metricsPreset.
%
% OPTIONS STRUCT FIELDS:
%   positiveClass (required) - Scalar label representing the positive class.
%   negativeClass (optional) - Scalar label for the negative class.
%   metricNames (optional)   - Cell array of metric names to compute.
%   metricsPreset (optional) - Name of a preset returned by
%                              metric_name_presets().
%
% OUTPUTS:
%   metrics                 - Struct containing calculated metrics. Each
%                             field corresponds to a name in metricNames.
%
% Date: 2025-05-15

function metrics = calculate_performance_metrics(y_true, y_pred, y_scores_positive_class, optionsOrPositive, metricNames)

    if nargin < 4
        error('calculate_performance_metrics requires at least four inputs.');
    end

    % Normalise options / inputs ------------------------------------------------
    opts = struct();
    if isstruct(optionsOrPositive)
        opts = optionsOrPositive;
    else
        opts.positiveClass = optionsOrPositive;
    end

    if ~isfield(opts, 'positiveClass') || isempty(opts.positiveClass)
        error('calculate_performance_metrics:MissingPositiveClass', ...
            'Positive class label must be supplied via options.positiveClass.');
    end

    if nargin < 5 || isempty(metricNames)
        if isfield(opts, 'metricNames') && ~isempty(opts.metricNames)
            metricNames = opts.metricNames;
        elseif isfield(opts, 'metricsPreset') && ~isempty(opts.metricsPreset)
            metricNames = metric_name_presets(opts.metricsPreset);
        else
            metricNames = metric_name_presets('default');
        end
    end
    metricNames = ensure_cellstr(metricNames);

    if ~isfield(opts, 'negativeClass') || isempty(opts.negativeClass)
        uniqueLabels = unique(y_true);
        uniqueLabels(uniqueLabels == opts.positiveClass) = [];
        if numel(uniqueLabels) == 1
            opts.negativeClass = uniqueLabels;
        elseif isempty(uniqueLabels)
            opts.negativeClass = NaN;
        else
            opts.negativeClass = uniqueLabels(1);
        end
    end

    % -------------------------------------------------------------------------
    metrics = struct();

    % Ensure y_true and y_pred are column vectors
    y_true = y_true(:);
    y_pred = y_pred(:);

    if isempty(y_true) || isempty(y_pred)
        warning('calculate_performance_metrics:EmptyInput', ...
            'Empty y_true or y_pred. Returning NaN for all metrics.');
        metrics = assign_nan_metrics(metricNames);
        return;
    end

    % Binarise counts relative to the supplied positive class
    positiveClassLabel = opts.positiveClass;
    negativeMask = (y_true ~= positiveClassLabel);

    TP = sum(y_true == positiveClassLabel & y_pred == positiveClassLabel);
    TN = sum(negativeMask & y_pred ~= positiveClassLabel);
    FP = sum(negativeMask & y_pred == positiveClassLabel);
    FN = sum(y_true == positiveClassLabel & y_pred ~= positiveClassLabel);

    numTotalSamples = TP + TN + FP + FN;
    if numTotalSamples == 0
        metrics = assign_nan_metrics(metricNames);
        return;
    end

    if any(strcmpi(metricNames, 'Accuracy'))
        metrics.Accuracy = safe_divide(TP + TN, numTotalSamples);
    end

    if any(strcmpi(metricNames, 'Sensitivity_WHO3'))
        metrics.Sensitivity_WHO3 = safe_divide(TP, TP + FN);
    end

    if any(strcmpi(metricNames, 'Specificity_WHO1'))
        metrics.Specificity_WHO1 = safe_divide(TN, TN + FP);
    end

    if any(strcmpi(metricNames, 'PPV_WHO3'))
        metrics.PPV_WHO3 = safe_divide(TP, TP + FP);
    end

    if any(strcmpi(metricNames, 'NPV_WHO1'))
        metrics.NPV_WHO1 = safe_divide(TN, TN + FN);
    end

    if any(strcmpi(metricNames, 'F1_WHO3'))
        precision_who3 = safe_divide(TP, TP + FP);
        recall_who3 = safe_divide(TP, TP + FN);
        metrics.F1_WHO3 = safe_divide(2 * (precision_who3 * recall_who3), precision_who3 + recall_who3);
    end

    if any(strcmpi(metricNames, 'F2_WHO3'))
        beta = 2;
        precision_who3 = safe_divide(TP, TP + FP);
        recall_who3 = safe_divide(TP, TP + FN);
        numerator = (1 + beta^2) * (precision_who3 * recall_who3);
        denominator = (beta^2 * precision_who3) + recall_who3;
        metrics.F2_WHO3 = safe_divide(numerator, denominator);
    end

    if any(strcmpi(metricNames, 'AUC'))
        if ~isempty(y_scores_positive_class) && numel(unique(y_true)) > 1 && ~all(isnan(y_scores_positive_class))
            y_true_logical_for_auc = (y_true == positiveClassLabel);
            try
                [~,~,~,AUC_val] = perfcurve(y_true_logical_for_auc, y_scores_positive_class, true);
                metrics.AUC = AUC_val;
            catch ME_auc
                warning('calculate_performance_metrics:AUCFailed', ...
                    'Could not calculate AUC: %s. Setting AUC to NaN.', ME_auc.message);
                metrics.AUC = NaN;
            end
        else
            metrics.AUC = NaN;
        end
    end

    % Ensure all requested metrics exist (assign NaN where missing)
    metrics = ensure_all_metrics(metrics, metricNames);
end

function metrics = assign_nan_metrics(metricNames)
    metrics = struct();
    for i = 1:numel(metricNames)
        metrics.(metricNames{i}) = NaN;
    end
end

function val = safe_divide(numerator, denominator)
    if denominator == 0
        val = NaN;
    else
        val = numerator / denominator;
    end
end

function metrics = ensure_all_metrics(metrics, metricNames)
    orderedMetrics = struct();
    for i = 1:numel(metricNames)
        name = metricNames{i};
        if ~isfield(metrics, name) || isnan(metrics.(name)) || isinf(metrics.(name))
            orderedMetrics.(name) = NaN;
        else
            orderedMetrics.(name) = metrics.(name);
        end
    end
    metrics = orderedMetrics;
end

function out = ensure_cellstr(val)
    if isstring(val)
        out = cellstr(val);
    elseif ischar(val)
        out = {val};
    elseif iscell(val)
        out = val;
    else
        out = {};
    end
end

function metricList = metric_name_presets(presetName)
%METRIC_NAME_PRESETS Convenience helper for shared metric configurations.
%
%   LIST = METRIC_NAME_PRESETS(NAME) returns the list of metric names stored
%   under NAME. Presets currently include:
%       - 'default'
%       - 'phase2_model_selection'
%       - 'phase3_final_evaluation'
%       - 'probe_level_summary'
%
%   Calling METRIC_NAME_PRESETS without arguments returns a struct whose
%   fields correspond to all available presets.

    persistent presetStructCached

    if isempty(presetStructCached)
        thisFileDir = fileparts(mfilename('fullpath'));
        projectRoot = fileparts(thisFileDir);
        cfg = load_run_configuration(projectRoot);
        presetStructCached = cfg.metricPresets;
    end

    if nargin < 1 || isempty(presetName)
        metricList = presetStructCached;
        return;
    end

    if isfield(presetStructCached, presetName)
        metricList = presetStructCached.(presetName);
    else
        metricList = presetStructCached.default;
    end
end
