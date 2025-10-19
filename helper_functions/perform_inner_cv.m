% perform_inner_cv.m
%
% Helper function to perform inner cross-validation for hyperparameter tuning.
% Date: 2025-05-15 (Updated with fscmrmr fix; Corrected to enforce fisherFeaturePercent)

function [bestHyperparams, bestOverallPerfMetrics, diagnostics] = perform_inner_cv(...
    X_inner_train_full, y_inner_train_full, probeIDs_inner_train_full, ...
    pipelineConfig, wavenumbers_original, numInnerFolds, metricNames, positiveClassLabel)

    paramGridCells = {};
    paramNames = {};
    diagnostics = init_diagnostics('perform_inner_cv');

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
        entry = log_pipeline_message('warning', 'perform_inner_cv', ...
            'Priority metric F2_WHO3 not found in metricNames. Using "%s" for optimisation.', metricNames{1});
        diagnostics = record_diagnostic(diagnostics, entry, [], 'warning');
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
        entry = log_pipeline_message('error', 'perform_inner_cv', ...
            'Insufficient data or label diversity for inner CV. Returning defaults.');
        diagnostics = record_diagnostic(diagnostics, entry, [], 'error');
        return;
    end

    totalInnerSteps = max(1, numel(hyperparamCombinations) * actualNumInnerFolds);
    innerReporter = ProgressReporter(progressLabel, totalInnerSteps, 'Verbose', verbose, 'ThrottleSeconds', 0);

    probe_WHO_Grade_inner = zeros(length(uniqueProbesInner), 1);
    [~, ~, groupIdxPerSpectrum_inner] = unique(probeIDs_inner_train_full, 'stable'); 
    for i = 1:length(uniqueProbesInner)
        probeSpectraLabels_inner = y_inner_train_full(groupIdxPerSpectrum_inner == i);
        if any(probeSpectraLabels_inner == positiveClassLabel)
            probe_WHO_Grade_inner(i) = positiveClassLabel;
        else
            probe_WHO_Grade_inner(i) = mode(probeSpectraLabels_inner);
        end
    end
    try
        innerCV_probeLevel = cvpartition(probe_WHO_Grade_inner, 'KFold', actualNumInnerFolds);
    catch ME_cvp
        entry = log_pipeline_message('warning', 'perform_inner_cv:cvpartition', ...
            'Stratified cvpartition failed (%s). Falling back to unstratified folds.', ME_cvp.message);
        diagnostics = record_diagnostic(diagnostics, entry, ME_cvp, 'warning');
        if actualNumInnerFolds >= 2 && length(probe_WHO_Grade_inner) >= actualNumInnerFolds
            try
                innerCV_probeLevel = cvpartition(length(probe_WHO_Grade_inner), 'KFold', actualNumInnerFolds);
            catch ME_cvp_unstrat
                entry = log_pipeline_message('error', 'perform_inner_cv:cvpartition', ...
                    'Failed to create inner CV folds: %s', ME_cvp_unstrat.message);
                diagnostics = record_diagnostic(diagnostics, entry, ME_cvp_unstrat, 'error');
                if ~isempty(hyperparamCombinations), bestHyperparams = hyperparamCombinations{1}; else, bestHyperparams = struct('binningFactor',1); end
                for iMet = 1:length(metricNames), bestOverallPerfMetrics.(metricNames{iMet}) = 0; end
                return;
            end
        else
            entry = log_pipeline_message('error', 'perform_inner_cv:cvpartition', ...
                'Not enough probes (%d) for %d folds.', length(probe_WHO_Grade_inner), actualNumInnerFolds);
            diagnostics = record_diagnostic(diagnostics, entry, [], 'error');
            if ~isempty(hyperparamCombinations), bestHyperparams = hyperparamCombinations{1}; else, bestHyperparams = struct('binningFactor',1); end
            for iMet = 1:length(metricNames), bestOverallPerfMetrics.(metricNames{iMet}) = 0; end
            return;
        end
    end

    numCombos = length(hyperparamCombinations);
    for iCombo = 1:numCombos
        currentHyperparams = hyperparamCombinations{iCombo};
        tempFoldMetricsArr = NaN(actualNumInnerFolds, length(metricNames));

        for kInner = 1:actualNumInnerFolds
            stepMsg = sprintf('Combo %d/%d, fold %d/%d', iCombo, numCombos, kInner, actualNumInnerFolds);
            updatedThisIteration = false;
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
                innerReporter.update(1, stepMsg + " - skipped");
                updatedThisIteration = true;
                continue;
            end

            [X_train_p, current_w_fold, preprocessInfo] = apply_pipeline_preprocessing( ...
                X_train_fold, wavenumbers_original, currentHyperparams);
            X_val_p = apply_pipeline_preprocessing( ...
                X_val_fold, wavenumbers_original, currentHyperparams, preprocessInfo);

            if currentHyperparams.binningFactor > 1
                [X_train_p, current_w_fold] = bin_spectra(X_train_fold, wavenumbers_original, currentHyperparams.binningFactor);
                [X_val_p, ~] = bin_spectra(X_val_fold, wavenumbers_original, currentHyperparams.binningFactor);
            end

            selectedFcIdx_in_current_w = 1:size(X_train_p, 2); 

            switch lower(pipelineConfig.feature_selection_method)
                case 'fisher'
                    % --- FIX STARTS HERE ---
                    % This block is simplified to only handle `fisherFeaturePercent`.
                    if isfield(currentHyperparams, 'fisherFeaturePercent')
                        numFeat = ceil(currentHyperparams.fisherFeaturePercent * size(X_train_p,2));
                    else
                        % This error indicates a mismatch between the pipeline definition
                        % and the logic here. It's better to fail fast than use wrong logic.
                        error('perform_inner_cv:MissingHyperparameter', ...
                              ['Fisher pipeline expects "fisherFeaturePercent" but it was not found. ', ...
                               'Troubleshooting tip: include fisherFeaturePercent in the hyperparameter ', ...
                               'grid for Fisher feature selection or review the pipeline configuration.']);
                    end
                    numFeat = min(numFeat, size(X_train_p,2));
                    % --- FIX ENDS HERE ---

                    if numFeat > 0 && size(X_train_p,1)>1 && length(unique(y_train_fold))==2
                        fisherRatios_inner = calculate_fisher_ratio(X_train_p, y_train_fold);
                        [~, sorted_idx_inner] = sort(fisherRatios_inner, 'descend', 'MissingPlacement','last');
                        selectedFcIdx_in_current_w = sorted_idx_inner(1:numFeat);
                    end
                case 'pca'
                    if size(X_train_p,2) > 0 && size(X_train_p,1) > 1 && size(X_train_p,1) > size(X_train_p,2) % N > P condition for standard PCA
                        try
                            cacheConfig = struct('signature', struct( ...
                                'context', 'perform_inner_cv', ...
                                'hyperparams', currentHyperparams));
                            [coeff_i, score_train_i, ~, ~, explained_i, mu_pca_i] = cached_pca(X_train_p, cacheConfig);
                            numComponents_i = 0;
                            if isfield(currentHyperparams, 'pcaVarianceToExplain')
                                cumulativeExplained_i = cumsum(explained_i);
                                idx_pc = find(cumulativeExplained_i >= currentHyperparams.pcaVarianceToExplain*100, 1, 'first');
                                if isempty(idx_pc), numComponents_i = size(coeff_i,2); else, numComponents_i = idx_pc; end
                            else 
                                numComponents_i = min(currentHyperparams.numPCAComponents, size(coeff_i,2));
                            end
                            
                            if numComponents_i > 0 && size(score_train_i,2) >= numComponents_i
                                X_train_p = score_train_i(:, 1:numComponents_i);
                                X_val_p = (X_val_p - mu_pca_i) * coeff_i(:, 1:numComponents_i);
                                selectedFcIdx_in_current_w = 1:numComponents_i; 
                            end
                        catch ME_pca_inner
                            entry = log_pipeline_message('warning', sprintf('perform_inner_cv:%s:PCA', pipelineConfig.name), ...
                                'PCA failed on inner fold %d (combo %d): %s', kInner, iCombo, ME_pca_inner.message);
                            diagnostics = record_diagnostic(diagnostics, entry, ME_pca_inner, 'warning');
                            selectedFcIdx_in_current_w = [];
                        end
                    end
                case 'mrmr'
                    numFeat = ceil(currentHyperparams.mrmrFeaturePercent * size(X_train_p,2));
                    numFeat = min(numFeat, size(X_train_p,2));

                    if numFeat <=0 || size(X_train_p,2) == 0 
                        selectedFcIdx_in_current_w = 1:size(X_train_p,2); 
                    elseif ~(size(X_train_p,1)>1 && length(unique(y_train_fold))==2 && exist('fscmrmr','file'))
                        selectedFcIdx_in_current_w = 1:size(X_train_p,2); % Fallback to all
                    else
                        try
                            y_train_fold_cat = categorical(y_train_fold);
                            if size(X_train_p,2) == 0 % No features in input X
                                selectedFcIdx_in_current_w = []; 
                            else
                                [ranked_indices_inner, ~] = fscmrmr(X_train_p, y_train_fold_cat); 
                                actual_num_to_take_inner = min(numFeat, length(ranked_indices_inner));
                                if actual_num_to_take_inner > 0
                                    selectedFcIdx_in_current_w = ranked_indices_inner(1:actual_num_to_take_inner);
                                else
                                    selectedFcIdx_in_current_w = [];
                                end
                            end
                        catch ME_mrmr_inner
                            entry = log_pipeline_message('warning', sprintf('perform_inner_cv:%s:MRMR', pipelineConfig.name), ...
                                'fscmrmr failed on inner fold %d (combo %d): %s', kInner, iCombo, ME_mrmr_inner.message);
                            diagnostics = record_diagnostic(diagnostics, entry, ME_mrmr_inner, 'warning');
                            selectedFcIdx_in_current_w = [];
                        end
                    end
            end
            
            if isempty(selectedFcIdx_in_current_w) && size(X_train_p, 2) > 0
                entry = log_pipeline_message('warning', sprintf('perform_inner_cv:%s:FeatureSelection', pipelineConfig.name), ...
                    'Empty feature set on inner fold %d (combo %d). Marking fold as invalid.', kInner, iCombo);
                diagnostics = record_diagnostic(diagnostics, entry, [], 'warning');
                tempFoldMetricsArr(kInner, :) = NaN;
                continue;
            elseif isempty(X_train_p) || size(X_train_p,2) == 0
                tempFoldMetricsArr(kInner, :) = NaN; continue;
            end

            [X_fs_train_fold, selectionInfo] = fit_pipeline_feature_selection( ...
                X_train_p, y_train_fold, pipelineConfig, currentHyperparams, current_w_fold);
            X_fs_val_fold = apply_pipeline_feature_selection(X_val_p, selectionInfo);

            classifier_inner = [];
            if isempty(X_fs_train_fold) || size(X_fs_train_fold,1)<2 || length(unique(y_train_fold))<2
                entry = log_pipeline_message('warning', sprintf('perform_inner_cv:%s:DataCheck', pipelineConfig.name), ...
                    'Insufficient samples or class diversity on inner fold %d (combo %d).', kInner, iCombo);
                diagnostics = record_diagnostic(diagnostics, entry, [], 'warning');
                tempFoldMetricsArr(kInner, :) = NaN; continue;
            end

            switch lower(pipelineConfig.classifier)
                case 'lda'
                    if size(X_fs_train_fold, 2) == 1 && var(X_fs_train_fold) < 1e-9
                        entry = log_pipeline_message('warning', sprintf('perform_inner_cv:%s:LDA', pipelineConfig.name), ...
                            'Singular feature detected on inner fold %d (combo %d).', kInner, iCombo);
                        diagnostics = record_diagnostic(diagnostics, entry, [], 'warning');
                        tempFoldMetricsArr(kInner, :) = NaN; continue;
                    end
                    try
                        classifier_inner = fitcdiscr(X_fs_train_fold, y_train_fold);
                    catch ME_lda_inner
                        entry = log_pipeline_message('warning', sprintf('perform_inner_cv:%s:LDA', pipelineConfig.name), ...
                            'LDA failed on inner fold %d (combo %d): %s', kInner, iCombo, ME_lda_inner.message);
                        diagnostics = record_diagnostic(diagnostics, entry, ME_lda_inner, 'warning');
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
                    entry = log_pipeline_message('warning', sprintf('perform_inner_cv:%s:Predict', pipelineConfig.name), ...
                        'Prediction or metric calculation failed on inner fold %d (combo %d): %s', kInner, iCombo, ME_predict_eval.message);
                    diagnostics = record_diagnostic(diagnostics, entry, ME_predict_eval, 'warning');
                    tempFoldMetricsArr(kInner, :) = NaN;
                end
            else
                tempFoldMetricsArr(kInner, :) = NaN;
            end

            if ~updatedThisIteration
                innerReporter.update(1, stepMsg);
                updatedThisIteration = true;
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
        entry = log_pipeline_message('error', sprintf('perform_inner_cv:%s', pipelineConfig.name), ...
            'No valid hyperparameter combination found. Defaulting to first combination.');
        diagnostics = record_diagnostic(diagnostics, entry, [], 'error');
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
    switch lower(level)
        case 'error'
            diagnostics.status = 'error';
        case 'warning'
            if ~strcmpi(diagnostics.status, 'error')
                diagnostics.status = 'warning';
            end
    end
end
