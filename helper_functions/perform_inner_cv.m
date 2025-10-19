% perform_inner_cv.m
%
% Helper function to perform inner cross-validation for hyperparameter tuning
% with the object-oriented pipeline API. The routine operates at the probe
% level to avoid data leakage between spectra originating from the same
% patient.
%
% Inputs:
%   X, y, probeIDs - training data restricted to the outer-CV training fold
%   pipelineConfig - pipelines.ClassificationPipeline instance
%   wavenumbers_original - row vector of wavenumbers for X
%   numInnerFolds - requested number of folds (adjusted if insufficient probes)
%   metricNames - cellstr of metric identifiers supported by
%                 calculate_performance_metrics
%   positiveClassLabel - label treated as positive class in metrics
%
% Name-value options:
%   'Verbose'       - logical flag controlling progress output (default true)
%   'ProgressLabel' - custom label for the ProgressReporter
%
% Outputs:
%   bestHyperparams        - struct containing the selected hyperparameters
%   bestOverallPerfMetrics - struct mapping metricNames to their mean scores
%   diagnostics            - struct capturing warnings/errors during CV
%
% Date: 2025-07-20

function [bestHyperparams, bestOverallPerfMetrics, diagnostics] = perform_inner_cv( ...
    X, y, probeIDs, pipelineConfig, wavenumbers_original, numInnerFolds, ...
    metricNames, positiveClassLabel, varargin)

    parser = inputParser();
    parser.FunctionName = 'perform_inner_cv';
    addParameter(parser, 'Verbose', true, @(v) islogical(v) || isnumeric(v));
    addParameter(parser, 'ProgressLabel', '', @(v) isstring(v) || ischar(v));
    parse(parser, varargin{:});

    verbose = logical(parser.Results.Verbose);
    progressLabel = strtrim(string(parser.Results.ProgressLabel));
    if strlength(progressLabel) == 0
        progressLabel = sprintf('Inner CV - %s', derive_pipeline_name(pipelineConfig));
    end

    diagnostics = init_diagnostics('perform_inner_cv');

    if nargin < 8 || isempty(metricNames)
        metricNames = {'Accuracy','Sensitivity_WHO3','Specificity_WHO1','PPV_WHO3', ...
            'NPV_WHO1','F1_WHO3','F2_WHO3','AUC'};
    end
    if nargin < 9 || isempty(positiveClassLabel)
        positiveClassLabel = 3;
    end

    if ~isa(pipelineConfig, 'pipelines.ClassificationPipeline')
        error('perform_inner_cv:UnsupportedPipeline', ...
            'pipelineConfig must be a pipelines.ClassificationPipeline instance.');
    end

    if iscolumn(wavenumbers_original)
        wavenumbers_original = wavenumbers_original';
    end

    hyperparamCombinations = pipelineConfig.getHyperparameterCombinations();
    if isempty(hyperparamCombinations)
        hyperparamCombinations = {struct()};
    end

    % Metric bookkeeping
    bestHyperparams = struct();
    bestOverallPerfMetrics = cell2struct(num2cell(nan(1, numel(metricNames))), metricNames, 2);
    priorityMetricName = select_priority_metric(metricNames);
    bestInnerPerfScore = -Inf;

    % Determine actual fold count based on probe availability
    uniqueProbes = unique(probeIDs, 'stable');
    if numel(uniqueProbes) < 1 || numel(unique(y)) < 2
        % Not enough data for CV; return defaults
        bestHyperparams = hyperparamCombinations{1};
        for iMet = 1:numel(metricNames)
            bestOverallPerfMetrics.(metricNames{iMet}) = 0;
        end
        entry = log_pipeline_message('error', 'perform_inner_cv:dataCheck', ...
            'Insufficient data or label diversity for inner CV. Returning defaults.');
        diagnostics = record_diagnostic(diagnostics, entry, [], 'error');
        return;
    end

    actualNumInnerFolds = min(numInnerFolds, numel(uniqueProbes));
    actualNumInnerFolds = max(2, actualNumInnerFolds);

    try
        stratLabels = assign_probe_level_labels(y, probeIDs, uniqueProbes, positiveClassLabel);
        innerCV = cvpartition(stratLabels, 'KFold', actualNumInnerFolds);
    catch ME_cvp
        entry = log_pipeline_message('warning', 'perform_inner_cv:cvpartition', ...
            'Stratified cvpartition failed (%s). Falling back to unstratified folds.', ME_cvp.message);
        diagnostics = record_diagnostic(diagnostics, entry, ME_cvp, 'warning');
        innerCV = cvpartition(numel(uniqueProbes), 'KFold', actualNumInnerFolds);
    end

    totalSteps = numel(hyperparamCombinations) * actualNumInnerFolds;
    progress = ProgressReporter(progressLabel, totalSteps, 'Verbose', verbose, 'ThrottleSeconds', 0);

    for iCombo = 1:numel(hyperparamCombinations)
        currentHyper = hyperparamCombinations{iCombo};
        foldMetrics = NaN(actualNumInnerFolds, numel(metricNames));

        for foldIdx = 1:actualNumInnerFolds
            stepMessage = sprintf('Combo %d/%d, fold %d/%d', iCombo, numel(hyperparamCombinations), foldIdx, actualNumInnerFolds);
            progress.update(1, stepMessage);

            trainProbeIdx = training(innerCV, foldIdx);
            valProbeIdx = test(innerCV, foldIdx);
            trainProbes = uniqueProbes(trainProbeIdx);
            valProbes = uniqueProbes(valProbeIdx);

            trainMask = ismember(probeIDs, trainProbes);
            valMask = ismember(probeIDs, valProbes);

            X_train = X(trainMask, :);
            y_train = y(trainMask);
            X_val = X(valMask, :);
            y_val = y(valMask);

            if isempty(X_train) || isempty(X_val) || numel(unique(y_train)) < 2
                foldMetrics(foldIdx, :) = NaN;
                continue;
            end

            try
                trainedPipeline = pipelineConfig.fit(X_train, y_train, wavenumbers_original, currentHyper);
            catch ME_fit
                entry = log_pipeline_message('warning', sprintf('perform_inner_cv:%s:fit', derive_pipeline_name(pipelineConfig)), ...
                    'Pipeline fitting failed on combo %d (fold %d): %s', iCombo, foldIdx, ME_fit.message);
                diagnostics = record_diagnostic(diagnostics, entry, ME_fit, 'warning');
                foldMetrics(foldIdx, :) = NaN;
                continue;
            end

            try
                [y_pred, scores, classNames] = trainedPipeline.predict(X_val, wavenumbers_original);
                metricsStruct = evaluate_pipeline_metrics(y_val, y_pred, scores, classNames, metricNames, positiveClassLabel);
                foldMetrics(foldIdx, :) = cellfun(@(mn) metricsStruct.(mn), metricNames);
            catch ME_eval
                entry = log_pipeline_message('warning', sprintf('perform_inner_cv:%s:evaluate', derive_pipeline_name(pipelineConfig)), ...
                    'Evaluation failed on combo %d (fold %d): %s', iCombo, foldIdx, ME_eval.message);
                diagnostics = record_diagnostic(diagnostics, entry, ME_eval, 'warning');
                foldMetrics(foldIdx, :) = NaN;
            end
        end

        meanMetrics = nanmean(foldMetrics, 1);
        if all(isnan(meanMetrics))
            continue;
        end

        priorityIdx = find(strcmpi(metricNames, priorityMetricName), 1);
        if isempty(priorityIdx)
            priorityIdx = 1;
        end
        priorityScore = meanMetrics(priorityIdx);
        if isnan(priorityScore)
            priorityScore = -Inf;
        end

        if priorityScore > bestInnerPerfScore
            bestInnerPerfScore = priorityScore;
            bestHyperparams = currentHyper;
            for iMet = 1:numel(metricNames)
                bestOverallPerfMetrics.(metricNames{iMet}) = meanMetrics(iMet);
            end
        end
    end

    if isempty(fieldnames(bestHyperparams)) && ~isempty(hyperparamCombinations)
        bestHyperparams = hyperparamCombinations{1};
        entry = log_pipeline_message('warning', sprintf('perform_inner_cv:%s', derive_pipeline_name(pipelineConfig)), ...
            'No valid hyperparameter combination succeeded. Defaulting to first combination.');
        diagnostics = record_diagnostic(diagnostics, entry, [], 'warning');
    end
end

function diagnostics = init_diagnostics(context)
    diagnostics = struct();
    diagnostics.status = 'ok';
    diagnostics.entries = struct('timestamp',{},'level',{},'context',{},'message',{});
    diagnostics.errors = {};
    diagnostics.context = context;
end

function diagnostics = record_diagnostic(diagnostics, entry, exceptionObj, level)
    diagnostics.entries(end+1) = entry; %#ok<AGROW>
    if nargin >= 3 && ~isempty(exceptionObj)
        diagnostics.errors{end+1} = exceptionObj; %#ok<AGROW>
    end
    if nargin < 4
        level = entry.level;
    end
    switch lower(string(level))
        case "error"
            diagnostics.status = 'error';
        case "warning"
            if ~strcmpi(diagnostics.status, 'error')
                diagnostics.status = 'warning';
            end
    end
end

function labels = assign_probe_level_labels(y, probeIDs, uniqueProbes, positiveClassLabel)
    labels = zeros(numel(uniqueProbes),1);
    for i = 1:numel(uniqueProbes)
        probeMask = ismember(probeIDs, uniqueProbes(i));
        probeLabels = y(probeMask);
        if any(probeLabels == positiveClassLabel)
            labels(i) = positiveClassLabel;
        else
            labels(i) = mode(probeLabels);
        end
    end
end

function name = derive_pipeline_name(pipelineConfig)
    if isa(pipelineConfig, 'pipelines.ClassificationPipeline')
        name = char(pipelineConfig.Name);
    else
        name = 'pipeline';
    end
end

function metricName = select_priority_metric(metricNames)
    defaultPriority = 'F2_WHO3';
    idx = find(strcmpi(metricNames, defaultPriority), 1);
    if ~isempty(idx)
        metricName = metricNames{idx};
        return;
    end
    metricName = metricNames{1};
end
