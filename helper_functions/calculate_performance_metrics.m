%% 
% 

% calculate_performance_metrics.m
%
% Helper function to calculate various classification performance metrics.
%
% INPUTS:
%   y_true                  - (N_samples x 1) True class labels.
%   y_pred                  - (N_samples x 1) Predicted class labels.
%   y_scores_positive_class - (N_samples x 1) Scores for the positive class (e.g., probabilities for WHO-3).
%                             Can be empty if AUC is not needed or not available.
%   positiveClassLabel      - Scalar. The label value representing the positive class (e.g., 3 for WHO-3).
%   metricNames             - Cell array of strings with names of metrics to compute.
%
% OUTPUTS:
%   metrics                 - Struct containing calculated metrics. Each field corresponds
%                             to a name in metricNames.
%
% Date: 2025-05-15

function metrics = calculate_performance_metrics(y_true, y_pred, y_scores_positive_class, positiveClassLabel, metricNames)

    metrics = struct();
    
    % Ensure y_true and y_pred are column vectors
    y_true = y_true(:);
    y_pred = y_pred(:);

    if isempty(y_true) || isempty(y_pred)
        warning('calculate_performance_metrics: Empty y_true or y_pred. Returning NaN for all metrics.');
        for i = 1:length(metricNames)
            metrics.(metricNames{i}) = NaN;
        end
        return;
    end
    
    % Define negative class label (assuming binary classification context here)
    % This assumes labels in y_true are only positiveClassLabel or one other label.
    all_labels = unique(y_true);
    negativeClassLabels = all_labels(all_labels ~= positiveClassLabel);
    if isempty(negativeClassLabels) && length(all_labels) == 1 && all_labels == positiveClassLabel
        % Only positive class present in y_true
        % TN and FP will be 0. Specificity might be NaN or 1 depending on definition.
        % This scenario means we can't calculate all metrics meaningfully if y_true contains only one class.
    elseif length(negativeClassLabels) > 1
        warning('More than one negative class label found. Metrics like Specificity_WHO1 might be ambiguous if "WHO1" isn''t the sole negative label.');
        % For now, proceed by considering anything not positiveClassLabel as negative.
    end


    TP = sum(y_true == positiveClassLabel & y_pred == positiveClassLabel);
    TN = sum(y_true ~= positiveClassLabel & y_pred ~= positiveClassLabel); % Assumes all non-positive are negative
    FP = sum(y_true ~= positiveClassLabel & y_pred == positiveClassLabel);
    FN = sum(y_true == positiveClassLabel & y_pred ~= positiveClassLabel);

    numTotalSamples = TP + TN + FP + FN;
    if numTotalSamples == 0 % Should have been caught by empty check, but good for robustness
        numTotalSamples = NaN; % To make metrics NaN
    end

    if any(strcmpi(metricNames, 'Accuracy'))
        metrics.Accuracy = (TP + TN) / numTotalSamples;
    end

    % Sensitivity (Recall or True Positive Rate) for the positive class (WHO-3)
    if any(strcmpi(metricNames, 'Sensitivity_WHO3'))
        metrics.Sensitivity_WHO3 = TP / (TP + FN);
    end

    % Specificity for the negative class (e.g., WHO-1) (True Negative Rate)
    if any(strcmpi(metricNames, 'Specificity_WHO1'))
        metrics.Specificity_WHO1 = TN / (TN + FP);
    end

    % Precision (Positive Predictive Value - PPV) for the positive class (WHO-3)
    if any(strcmpi(metricNames, 'PPV_WHO3'))
        metrics.PPV_WHO3 = TP / (TP + FP);
    end

    % Negative Predictive Value (NPV)
    if any(strcmpi(metricNames, 'NPV_WHO1')) % Assuming WHO1 is the negative concept here
        metrics.NPV_WHO1 = TN / (TN + FN);
    end

    % F1-Score for the positive class (WHO-3)
    if any(strcmpi(metricNames, 'F1_WHO3'))
        precision_who3 = TP / (TP + FP); % Re-calculate to handle local scope if PPV_WHO3 not requested
        recall_who3 = TP / (TP + FN);    % Re-calculate
        metrics.F1_WHO3 = 2 * (precision_who3 * recall_who3) / (precision_who3 + recall_who3);
    end

    % F2-Score for the positive class (WHO-3) (weights recall higher)
    if any(strcmpi(metricNames, 'F2_WHO3'))
        beta = 2;
        precision_who3 = TP / (TP + FP); % Re-calculate
        recall_who3 = TP / (TP + FN);    % Re-calculate
        metrics.F2_WHO3 = (1 + beta^2) * (precision_who3 * recall_who3) / ((beta^2 * precision_who3) + recall_who3);
    end
    
    % AUC (Area Under ROC Curve)
    if any(strcmpi(metricNames, 'AUC'))
        if ~isempty(y_scores_positive_class) && length(unique(y_true)) > 1
            y_true_logical_for_auc = (y_true == positiveClassLabel); % Convert to logical (true for positive class)
            try
                [~,~,~,AUC_val] = perfcurve(y_true_logical_for_auc, y_scores_positive_class, true); % 'true' is the positive class indicator in logical array
                metrics.AUC = AUC_val;
            catch ME_auc
                fprintf('Warning: Could not calculate AUC: %s. Setting AUC to NaN.\n', ME_auc.message);
                metrics.AUC = NaN;
            end
        else
            if length(unique(y_true)) <= 1
                % AUC is not well-defined if only one class is present in y_true
                % fprintf('Note: AUC not calculated because y_true contains only one class.\n');
            end
            metrics.AUC = NaN;
        end
    end
    
    % Ensure all requested metrics are present in the output struct, assign NaN if undefined (e.g. div by zero)
    for i = 1:length(metricNames)
        if isfield(metrics, metricNames{i})
            if isnan(metrics.(metricNames{i})) || isinf(metrics.(metricNames{i}))
                % Ensure NaNs for ill-defined cases (e.g. 0/0), but allow 0 for cases like 0 / (non-zero)
                % If TP+FN = 0, Sensitivity = TP/(TP+FN) = 0/0 = NaN.
                % If TP+FP = 0, Precision = TP/(TP+FP) = 0/0 = NaN.
                % F-scores will also be NaN if precision or recall is NaN.
                % If a metric is Inf (e.g. a perfect score where denominator component is zero, like 1/0 for Recall if TP>0, FN=0, but TP+FN=0 means TP=0,FN=0)
                % this should ideally be handled by the calculation itself being 1 or 0.
                % The current structure results in NaN for 0/0. Inf is less likely with these specific formulas.
            end
        else
            metrics.(metricNames{i}) = NaN; % Metric was requested but not computed by any if-block
        end
    end
end