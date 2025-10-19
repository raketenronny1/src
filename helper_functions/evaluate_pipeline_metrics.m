function [metrics, positiveScores] = evaluate_pipeline_metrics(y_true, y_pred, scoreData, classNames, metricNames, positiveClassLabel)
%EVALUATE_PIPELINE_METRICS Wrapper around calculate_performance_metrics.
%   METRICS = EVALUATE_PIPELINE_METRICS(Y_TRUE, Y_PRED, SCOREDATA, CLASSNAMES,
%   METRICNAMES, POSLABEL) extracts the positive-class scores and invokes
%   CALCULATE_PERFORMANCE_METRICS with a consistent metric list and positive
%   class label handling. SCOREDATA can be either a column vector of positive
%   class scores or the full score matrix returned by a classifier.
%
%   If CLASSNAMES is provided together with a score matrix, the helper
%   locates the column that corresponds to POSLABEL (default WHO-3 / value 3).
%   When CLASSNAMES is empty, SCOREDATA is assumed to already correspond to
%   the positive-class scores.
%
%   Date: 2025-07-07

    if nargin < 4
        classNames = [];
    end
    if nargin < 5 || isempty(metricNames)
        metricNames = {'Accuracy','Sensitivity_WHO3','Specificity_WHO1', ...
            'PPV_WHO3','NPV_WHO1','F1_WHO3','F2_WHO3','AUC'};
    end
    if nargin < 6 || isempty(positiveClassLabel)
        positiveClassLabel = 3;
    end

    positiveScores = [];
    if isempty(scoreData)
        positiveScores = [];
    elseif isempty(classNames) || isscalar(scoreData) || size(scoreData,2) == 1
        positiveScores = scoreData(:);
    else
        idx = locate_positive_class_index(classNames, positiveClassLabel);
        if isempty(idx) || idx > size(scoreData,2)
            warning('evaluate_pipeline_metrics:MissingPositiveClass', ...
                'Positive class %s not found in provided classNames. Returning NaN for score-based metrics.', num2str(positiveClassLabel));
            positiveScores = [];
        else
            positiveScores = scoreData(:, idx);
        end
    end

    metrics = calculate_performance_metrics(y_true, y_pred, positiveScores, positiveClassLabel, metricNames);
end

function idx = locate_positive_class_index(classNames, positiveClassLabel)
    if isa(classNames, 'categorical')
        classNames = string(classNames);
    end
    if iscell(classNames)
        classNames = string(classNames);
    end
    if isnumeric(classNames)
        idx = find(classNames == positiveClassLabel, 1, 'first');
        return;
    end
    classStr = string(classNames);
    posStr = string(positiveClassLabel);
    idx = find(classStr == posStr, 1, 'first');
    if isempty(idx) && all(~isnan(str2double(classStr)))
        idx = find(str2double(classStr) == positiveClassLabel, 1, 'first');
    end
end
