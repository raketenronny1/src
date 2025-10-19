% perform_inner_cv.m
%
% Helper function to perform inner cross-validation for hyperparameter tuning.
% Date: 2025-05-15 (Updated with fscmrmr fix; Corrected to enforce fisherFeaturePercent)

function [bestHyperparams, bestOverallPerfMetrics] = perform_inner_cv(...
    X_inner_train_full, y_inner_train_full, probeIDs_inner_train_full, ...
    pipelineConfig, wavenumbers_original, numInnerFolds, metricNames)

    paramGridCells = {}; 
    paramNames = {};     

    if ismember('binningFactor', pipelineConfig.hyperparameters_to_tune)
        paramGridCells{end+1} = pipelineConfig.binningFactors(:)'; 
        paramNames{end+1} = 'binningFactor';
    end

    switch lower(pipelineConfig.feature_selection_method)
        case 'fisher'
            if ismember('fisherFeaturePercent', pipelineConfig.hyperparameters_to_tune)
                paramGridCells{end+1} = pipelineConfig.fisherFeaturePercent_range(:)';
                paramNames{end+1} = 'fisherFeaturePercent';
            end
        case 'pca'
            if ismember('pcaVarianceToExplain', pipelineConfig.hyperparameters_to_tune)
                paramGridCells{end+1} = pipelineConfig.pcaVarianceToExplain_range(:)'; 
                paramNames{end+1} = 'pcaVarianceToExplain';
            elseif ismember('numPCAComponents', pipelineConfig.hyperparameters_to_tune) 
                 paramGridCells{end+1} = pipelineConfig.numPCAComponents_range(:)'; 
                 paramNames{end+1} = 'numPCAComponents';
            end
        case 'mrmr'
            if ismember('mrmrFeaturePercent', pipelineConfig.hyperparameters_to_tune)
                paramGridCells{end+1} = pipelineConfig.mrmrFeaturePercent_range(:)';
                paramNames{end+1} = 'mrmrFeaturePercent';
            end
    end
    
    if isempty(paramGridCells)
        currentHyperparams = struct();
        if isfield(pipelineConfig,'binningFactors') && ~isempty(pipelineConfig.binningFactors) && ...
           ~ismember('binningFactor', pipelineConfig.hyperparameters_to_tune) 
             currentHyperparams.binningFactor = pipelineConfig.binningFactors(1); 
        elseif ~ismember('binningFactor', pipelineConfig.hyperparameters_to_tune)
             currentHyperparams.binningFactor = 1;
        end
        hyperparamCombinations = {currentHyperparams};
    else
        gridOutputs = cell(1, length(paramGridCells));
        [gridOutputs{:}] = ndgrid(paramGridCells{:}); 
        numCombinations = numel(gridOutputs{1});
        hyperparamCombinations = cell(numCombinations, 1);
        for iCombo = 1:numCombinations
            comboStruct = struct();
            for iParam = 1:length(paramNames)
                comboStruct.(paramNames{iParam}) = gridOutputs{iParam}(iCombo);
            end
             if ~isfield(comboStruct, 'binningFactor') 
                if isfield(pipelineConfig,'binningFactors') && ~isempty(pipelineConfig.binningFactors) && ...
                   ~ismember('binningFactor', pipelineConfig.hyperparameters_to_tune)
                    comboStruct.binningFactor = pipelineConfig.binningFactors(1); 
                elseif ~isfield(pipelineConfig,'binningFactors') 
                    comboStruct.binningFactor = 1; 
                end
            end
            hyperparamCombinations{iCombo} = comboStruct;
        end
    end
    
    priorityMetricName = 'F2_WHO3'; 
    f2_idx_in_metrics = find(strcmpi(metricNames, priorityMetricName));
    if isempty(f2_idx_in_metrics)
        warning('perform_inner_cv: Priority metric F2_WHO3 not found in metricNames. Using first metric "%s" for optimization.', metricNames{1});
        f2_idx_in_metrics = 1; 
        priorityMetricName = metricNames{1};
    end
    
    bestInnerPerfScore = -Inf; 
    bestHyperparams = struct();
    bestOverallPerfMetrics = struct(); 
    for iMet = 1:length(metricNames) 
        bestOverallPerfMetrics.(metricNames{iMet}) = NaN;
    end

    actualNumInnerFolds = numInnerFolds;
    uniqueProbesInner = unique(probeIDs_inner_train_full);
    if length(uniqueProbesInner) < numInnerFolds && length(uniqueProbesInner) > 0 
        actualNumInnerFolds = max(2, length(uniqueProbesInner)); 
    elseif isempty(uniqueProbesInner) && ~isempty(X_inner_train_full) 
         actualNumInnerFolds = 1; 
    end
    
    if isempty(X_inner_train_full) || length(unique(y_inner_train_full)) < 2 || actualNumInnerFolds < 2
        if ~isempty(hyperparamCombinations)
            bestHyperparams = hyperparamCombinations{1}; 
        else 
            bestHyperparams = struct('binningFactor',1); 
            if isfield(pipelineConfig,'binningFactors') && ~isempty(pipelineConfig.binningFactors)
                 bestHyperparams.binningFactor = pipelineConfig.binningFactors(1);
            end
        end
        for iMet = 1:length(metricNames), bestOverallPerfMetrics.(metricNames{iMet}) = 0; end
        return;
    end

    probe_WHO_Grade_inner = zeros(length(uniqueProbesInner), 1);
    [~, ~, groupIdxPerSpectrum_inner] = unique(probeIDs_inner_train_full, 'stable'); 
    for i = 1:length(uniqueProbesInner)
        probeSpectraLabels_inner = y_inner_train_full(groupIdxPerSpectrum_inner == i);
        if any(probeSpectraLabels_inner == 3) 
            probe_WHO_Grade_inner(i) = 3;
        else
            probe_WHO_Grade_inner(i) = mode(probeSpectraLabels_inner); 
        end
    end
    try
        innerCV_probeLevel = cvpartition(probe_WHO_Grade_inner, 'KFold', actualNumInnerFolds);
    catch ME_cvp
        if actualNumInnerFolds >= 2 && length(probe_WHO_Grade_inner) >= actualNumInnerFolds
            try 
                innerCV_probeLevel = cvpartition(length(probe_WHO_Grade_inner), 'KFold', actualNumInnerFolds); 
            catch 
                 if ~isempty(hyperparamCombinations), bestHyperparams = hyperparamCombinations{1}; else, bestHyperparams = struct('binningFactor',1); end
                 for iMet = 1:length(metricNames), bestOverallPerfMetrics.(metricNames{iMet}) = 0; end
                 return;
            end
        else
            if ~isempty(hyperparamCombinations), bestHyperparams = hyperparamCombinations{1}; else, bestHyperparams = struct('binningFactor',1); end
            for iMet = 1:length(metricNames), bestOverallPerfMetrics.(metricNames{iMet}) = 0; end
            return;
        end
    end

    for iCombo = 1:length(hyperparamCombinations)
        currentHyperparams = hyperparamCombinations{iCombo};
        tempFoldMetricsArr = NaN(actualNumInnerFolds, length(metricNames));

        for kInner = 1:actualNumInnerFolds
            isInnerTrainProbe_idx = training(innerCV_probeLevel, kInner);
            isInnerValProbe_idx   = test(innerCV_probeLevel, kInner);

            innerTrainProbeIDs = uniqueProbesInner(isInnerTrainProbe_idx);
            innerValProbeIDs   = uniqueProbesInner(isInnerValProbe_idx);

            idxInnerTrain_Spectra = ismember(probeIDs_inner_train_full, innerTrainProbeIDs);
            idxInnerVal_Spectra   = ismember(probeIDs_inner_train_full, innerValProbeIDs);

            X_train_fold = X_inner_train_full(idxInnerTrain_Spectra, :);
            y_train_fold = y_inner_train_full(idxInnerTrain_Spectra);
            X_val_fold   = X_inner_train_full(idxInnerVal_Spectra, :);
            y_val_fold   = y_inner_train_full(idxInnerVal_Spectra);
            
            if isempty(X_train_fold) || isempty(X_val_fold) || length(unique(y_train_fold))<2
                tempFoldMetricsArr(kInner, :) = NaN; 
                continue;
            end

            [X_train_p, current_w_fold, preprocessInfo] = apply_pipeline_preprocessing( ...
                X_train_fold, wavenumbers_original, currentHyperparams);
            X_val_p = apply_pipeline_preprocessing( ...
                X_val_fold, wavenumbers_original, currentHyperparams, preprocessInfo);

            if isempty(X_train_p) || size(X_train_p,2) == 0
                tempFoldMetricsArr(kInner, :) = NaN; continue;
            end

            [X_fs_train_fold, selectionInfo] = fit_pipeline_feature_selection( ...
                X_train_p, y_train_fold, pipelineConfig, currentHyperparams, current_w_fold);
            X_fs_val_fold = apply_pipeline_feature_selection(X_val_p, selectionInfo);

            classifier_inner = [];
            if isempty(X_fs_train_fold) || size(X_fs_train_fold,1)<2 || length(unique(y_train_fold))<2
                tempFoldMetricsArr(kInner, :) = NaN; continue;
            end

            switch lower(pipelineConfig.classifier)
                case 'lda'
                    if size(X_fs_train_fold, 2) == 1 && var(X_fs_train_fold) < 1e-9 
                        tempFoldMetricsArr(kInner, :) = NaN; continue;
                    end
                    try
                        classifier_inner = fitcdiscr(X_fs_train_fold, y_train_fold);
                    catch ME_lda_inner
                        tempFoldMetricsArr(kInner, :) = NaN; continue;
                    end
            end

            if ~isempty(classifier_inner) && ~isempty(X_fs_val_fold) && ~isempty(y_val_fold)
                try
                    [y_pred_inner, y_scores_inner] = predict(classifier_inner, X_fs_val_fold);
                    currentInnerFoldMetricsStruct = evaluate_pipeline_metrics( ...
                        y_val_fold, y_pred_inner, y_scores_inner, classifier_inner.ClassNames, metricNames);
                    tempFoldMetricsArr(kInner, :) = cellfun(@(mn) currentInnerFoldMetricsStruct.(mn), metricNames);
                catch ME_predict_eval
                    tempFoldMetricsArr(kInner, :) = NaN;
                end
            else
                tempFoldMetricsArr(kInner, :) = NaN;
            end
        end 
        
        meanPerfThisCombo = nanmean(tempFoldMetricsArr, 1);
        
        currentComboPriorityScore = meanPerfThisCombo(f2_idx_in_metrics);
        if isnan(currentComboPriorityScore), currentComboPriorityScore = -Inf; end

        if currentComboPriorityScore > bestInnerPerfScore
            bestInnerPerfScore = currentComboPriorityScore;
            bestHyperparams = currentHyperparams;
            for iMet = 1:length(metricNames)
                if iMet <= length(meanPerfThisCombo) 
                    bestOverallPerfMetrics.(metricNames{iMet}) = meanPerfThisCombo(iMet);
                else
                    bestOverallPerfMetrics.(metricNames{iMet}) = NaN; 
                end
            end
        end
    end 
    
    if isempty(fieldnames(bestHyperparams)) && ~isempty(hyperparamCombinations) 
        bestHyperparams = hyperparamCombinations{1}; 
        for iMet=1:length(metricNames), bestOverallPerfMetrics.(metricNames{iMet}) = 0; end
        if isfield(bestOverallPerfMetrics, priorityMetricName) && ~isempty(f2_idx_in_metrics) % check f2_idx_in_metrics also
             bestOverallPerfMetrics.(metricNames{f2_idx_in_metrics}) = -Inf; % Ensure it doesn't look like a good result if all failed
        end
    end
end