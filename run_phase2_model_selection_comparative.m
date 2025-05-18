% run_phase2_model_selection_comparative.m
%
% Main script for Phase 2: Model and Feature Selection Pipelines.
% Implements nested cross-validation to find the best combination of
% preprocessing, feature selection, and classifier.
% THIS VERSION COMPARES TWO OUTLIER REMOVAL STRATEGIES ('OR' and 'AND')
% IN A SINGLE RUN and generates comparative plots.
%
% Date: 2025-05-18 (Modified for comprehensive strategy comparison)

%% 0. Initialization
fprintf('PHASE 2: Model Selection & Outlier Strategy Comparison - %s\n', string(datetime('now')));
clear; clc; close all;

% --- Define Paths ---
projectRoot = pwd; 
if ~exist(fullfile(projectRoot, 'src'), 'dir') || ~exist(fullfile(projectRoot, 'data'), 'dir')
    error('Project structure not found. Run from project root. Current: %s', projectRoot);
end

srcPath       = fullfile(projectRoot, 'src');
helperFunPath = fullfile(srcPath, 'helper_functions');
if ~exist(helperFunPath, 'dir'), error('Helper functions directory not found: %s', helperFunPath); end
addpath(helperFunPath);

dataPath      = fullfile(projectRoot, 'data');
resultsPath   = fullfile(projectRoot, 'results', 'Phase2'); 
modelsPath    = fullfile(projectRoot, 'models', 'Phase2');   
figuresPath   = fullfile(projectRoot, 'figures', 'Phase2'); 
comparisonFiguresPath = fullfile(projectRoot, 'figures', 'OutlierStrategyComparison'); % For new comparison plots
comparisonResultsPath = fullfile(projectRoot, 'results', 'OutlierStrategyComparison'); % For new comparison tables/data

dirToEnsure = {resultsPath, modelsPath, figuresPath, comparisonFiguresPath, comparisonResultsPath};
for i=1:length(dirToEnsure), if ~isfolder(dirToEnsure{i}), mkdir(dirToEnsure{i}); end, end

dateStrForFilenames = string(datetime('now','Format','yyyyMMdd'));

% --- Define Outlier Strategies to Compare & Result Storage ---
outlierStrategiesToCompare = {'OR', 'AND'};
overallComparisonResults = struct(); 

%% --- MAIN LOOP FOR OUTLIER STRATEGIES ---
for iStrategy = 1:length(outlierStrategiesToCompare)
    currentOutlierStrategy = outlierStrategiesToCompare{iStrategy};
    fprintf('\n\n====================================================================\n');
    fprintf('   PROCESSING WITH OUTLIER STRATEGY: T2 %s Q\n', currentOutlierStrategy);
    fprintf('====================================================================\n');

    %% 1. Load Data (Specific to currentOutlierStrategy)
    fprintf('\n--- 1. Loading Data (Outlier Strategy: %s) ---\n', currentOutlierStrategy);

    if strcmpi(currentOutlierStrategy, 'OR')
        % Use the most recent T2orQ file
        orFiles = dir(fullfile(dataPath, '*_training_set_no_outliers_T2orQ.mat'));
        if isempty(orFiles), error('No training set file found for OR strategy in %s.', dataPath); end
        [~,sortIdx] = sort([orFiles.datenum], 'descend');
        inputDataFile = fullfile(dataPath, orFiles(sortIdx(1)).name);
        fprintf('Loading OR strategy data from: %s\n', inputDataFile);
        loadedData = load(inputDataFile, ...
                           'X_train_no_outliers_OR', 'y_train_no_outliers_OR_num', ...
                           'Patient_ID_no_outliers_OR', 'wavenumbers_roi'); % Adjusted variable names
        X_train_full = loadedData.X_train_no_outliers_OR;
        y_train_full = loadedData.y_train_no_outliers_OR_num;
        probeIDs_train_full = loadedData.Patient_ID_no_outliers_OR; % Changed from Patient_ID_train_no_outliers_OR
        
    elseif strcmpi(currentOutlierStrategy, 'AND')
        % Use the most recent T2andQ file
        andFiles = dir(fullfile(dataPath, '*_training_set_no_outliers_T2andQ.mat'));
        if isempty(andFiles), error('No training set file found for AND strategy in %s.', dataPath); end
        [~,sortIdx] = sort([andFiles.datenum], 'descend');
        inputDataFile = fullfile(dataPath, andFiles(sortIdx(1)).name);
        fprintf('Loading AND strategy data from: %s\n', inputDataFile);
        loadedData = load(inputDataFile, ...
                           'X_train_no_outliers_AND', 'y_train_no_outliers_AND_num', ...
                           'Patient_ID_no_outliers_AND', 'wavenumbers_roi'); % Adjusted variable names
        X_train_full = loadedData.X_train_no_outliers_AND;
        y_train_full = loadedData.y_train_no_outliers_AND_num;
        probeIDs_train_full = loadedData.Patient_ID_no_outliers_AND; % Changed from Patient_ID_train_no_outliers_AND
    else
        error('Internal error: Invalid outlierStrategy in loop.');
    end
    
    if isfield(loadedData, 'wavenumbers_roi')
        wavenumbers_original = loadedData.wavenumbers_roi;
    else
        fprintf('wavenumbers_roi not found in strategy-specific file, loading from general wavenumbers.mat\n');
        w_data = load(fullfile(dataPath, 'wavenumbers.mat'), 'wavenumbers_roi');
        wavenumbers_original = w_data.wavenumbers_roi;
    end
    if iscolumn(wavenumbers_original), wavenumbers_original = wavenumbers_original'; end
    
    if isempty(X_train_full) || isempty(y_train_full) || isempty(probeIDs_train_full)
        error('Data loading failed for strategy %s: X, y, or probeIDs are empty.', currentOutlierStrategy);
    end
    if ~isnumeric(y_train_full)
         error('y_train_full is not numeric for strategy %s.', currentOutlierStrategy);
    end
    fprintf('Data for strategy %s loaded: %d spectra (%d unique probes), %d features.\n', ...
        currentOutlierStrategy, size(X_train_full, 1), length(unique(probeIDs_train_full)), size(X_train_full, 2));

    %% 2. Define Cross-Validation Parameters
    numOuterFolds = 5;
    rng('default'); % Reset RNG for some consistency if data inputs are similar

    [uniqueProbes, ~, groupIdxPerSpectrum] = unique(probeIDs_train_full, 'stable');
    if length(uniqueProbes) < numOuterFolds
        error('Strategy %s: Number of unique probes (%d) is less than numOuterFolds (%d). Reduce folds or check data.', ...
            currentOutlierStrategy, length(uniqueProbes), numOuterFolds);
    end
    probe_WHO_Grade = zeros(length(uniqueProbes), 1);
    for i_probe_cv = 1:length(uniqueProbes)
        probeSpectraLabels = y_train_full(groupIdxPerSpectrum == i_probe_cv);
        if any(probeSpectraLabels == 3), probe_WHO_Grade(i_probe_cv) = 3;
        else, probe_WHO_Grade(i_probe_cv) = mode(probeSpectraLabels); end
    end
    
    try
        outerCV_probeLevel = cvpartition(probe_WHO_Grade, 'KFold', numOuterFolds);
    catch ME_cv
        fprintf('Warning: Stratified CV failed for strategy %s (Error: %s). Attempting unstratified.\n', currentOutlierStrategy, ME_cv.message);
        try
            outerCV_probeLevel = cvpartition(length(uniqueProbes), 'KFold', numOuterFolds);
        catch ME_cv_un
            error('Could not create outer CV partition for strategy %s: %s', currentOutlierStrategy, ME_cv_un.message);
        end
    end

    numInnerFolds = 3; 
    metricNames = {'Accuracy', 'Sensitivity_WHO3', 'Specificity_WHO1', 'PPV_WHO3', 'NPV_WHO1', 'F1_WHO3', 'F2_WHO3', 'AUC'};

    %% 3. Define Pipelines to Evaluate (defined once, outside strategy loop if identical)
    if iStrategy == 1 % Define pipelines only on the first iteration
        pipelines = cell(0,1); 
        pipelineIdx = 0;
        % --- Pipeline 1: Baseline (Binning) + LDA ---
        p = struct(); p.name = 'BaselineLDA'; p.feature_selection_method = 'none'; p.classifier = 'LDA';
        p.hyperparameters_to_tune = {'binningFactor'}; p.binningFactors = [1, 2, 4, 8, 16];
        pipelineIdx = pipelineIdx + 1; pipelines{pipelineIdx} = p;
        % --- Pipeline 2: Fisher Ratio + LDA ---
        p = struct(); p.name = 'FisherLDA'; p.feature_selection_method = 'fisher'; p.classifier = 'LDA';
        p.hyperparameters_to_tune = {'binningFactor', 'numFisherFeatures'};
        p.binningFactors = [1, 2, 4, 8, 16]; p.numFisherFeatures_range = [10, 20, 30, 40, 50, 75, 100];
        pipelineIdx = pipelineIdx + 1; pipelines{pipelineIdx} = p;
        % --- Pipeline 3: PCA + LDA ---
        p = struct(); p.name = 'PCALDA'; p.feature_selection_method = 'pca'; p.classifier = 'LDA';
        p.hyperparameters_to_tune = {'binningFactor', 'pcaVarianceToExplain'};
        p.binningFactors = [1, 2, 4, 8, 16]; p.pcaVarianceToExplain_range = [0.90, 0.95, 0.99];
        pipelineIdx = pipelineIdx + 1; pipelines{pipelineIdx} = p;
        % --- Pipeline 4: MRMR + LDA ---
        p = struct(); p.name = 'MRMRLDA'; p.feature_selection_method = 'mrmr'; p.classifier = 'LDA';
        p.hyperparameters_to_tune = {'binningFactor', 'numMRMRFeatures'};
        p.binningFactors = [1, 2, 4, 8, 16]; p.numMRMRFeatures_range = [10, 20, 30, 40, 50];
        pipelineIdx = pipelineIdx + 1; pipelines{pipelineIdx} = p;
        
        overallComparisonResults.pipelines = pipelines; % Store pipeline definitions once
        overallComparisonResults.metricNames = metricNames; % Store metric names once
    end
    
    currentStrategyPipelinesResults = cell(length(pipelines), 1); % Results for this strategy

    %% 4. Nested Cross-Validation Loop (for currentOutlierStrategy)
    fprintf('\nStarting Nested Cross-Validation for Strategy: %s...\n', currentOutlierStrategy);

    for iPipeline = 1:length(pipelines)
        currentPipeline = pipelines{iPipeline};
        fprintf('\n  --- Evaluating Pipeline: %s (Strategy: %s) ---\n', currentPipeline.name, currentOutlierStrategy);

        outerFoldMetrics = NaN(numOuterFolds, length(metricNames));
        outerFoldBestHyperparams = cell(numOuterFolds, 1);
        % outerFoldModels = cell(numOuterFolds, 1); % Optional to save models
        % outerFoldSelectedFeaturesInfo = cell(numOuterFolds, 1); % Optional

        for kOuter = 1:numOuterFolds
            fprintf('    Outer Fold %d/%d:\n', kOuter, numOuterFolds);

            isOuterTrainProbe_IndicesInUniqueList = training(outerCV_probeLevel, kOuter);
            isOuterTestProbe_IndicesInUniqueList  = test(outerCV_probeLevel, kOuter);

            outerTrainProbeIDs = uniqueProbes(isOuterTrainProbe_IndicesInUniqueList);
            outerTestProbeIDs  = uniqueProbes(isOuterTestProbe_IndicesInUniqueList);

            idxOuterTrain = ismember(probeIDs_train_full, outerTrainProbeIDs);
            idxOuterTest  = ismember(probeIDs_train_full, outerTestProbeIDs);

            X_outer_train = X_train_full(idxOuterTrain, :);
            y_outer_train = y_train_full(idxOuterTrain); % numeric labels
            probeIDs_outer_train = probeIDs_train_full(idxOuterTrain);

            X_outer_test  = X_train_full(idxOuterTest, :);
            y_outer_test  = y_train_full(idxOuterTest); % numeric labels

            fprintf('      Outer train: %d spectra from %d probes. Outer test: %d spectra from %d probes.\n', ...
                size(X_outer_train,1), length(outerTrainProbeIDs), ...
                size(X_outer_test,1), length(outerTestProbeIDs));
            
            if isempty(X_outer_train) || isempty(X_outer_test)
                fprintf('      WARNING: Outer fold %d has empty training or test set. Skipping fold.\n', kOuter);
                outerFoldMetrics(kOuter, :) = NaN;
                continue;
            end
            if length(unique(y_outer_train)) < 2
                 fprintf('      WARNING: Outer fold %d training data has only one class. Skipping fold.\n', kOuter);
                 outerFoldMetrics(kOuter, :) = NaN;
                 continue;
            end

            % Call perform_inner_cv (ensure y_outer_train is numeric for it)
            [bestHyperparams, bestInnerPerf] = perform_inner_cv(...
                X_outer_train, y_outer_train, probeIDs_outer_train, ...
                currentPipeline, wavenumbers_original, numInnerFolds, metricNames);

            fprintf('      Best hyperparameters from inner CV: '); disp(bestHyperparams);
             if isstruct(bestInnerPerf) && isfield(bestInnerPerf, 'F2_WHO3')
                 fprintf('      Corresponding inner CV performance (F2_WHO3): %.4f\n', bestInnerPerf.F2_WHO3);
            else
                 fprintf('      Inner CV performance could not be determined or F2_WHO3 not available.\n');
            end
            outerFoldBestHyperparams{kOuter} = bestHyperparams;

            % --- Retrain model on X_outer_train with bestHyperparams and evaluate on X_outer_test ---
            currentWavenumbers_fold = wavenumbers_original;
            X_train_processed = X_outer_train;
            X_test_processed = X_outer_test;  

            if isfield(bestHyperparams, 'binningFactor') && bestHyperparams.binningFactor > 1
                [X_train_processed, currentWavenumbers_fold] = bin_spectra(X_outer_train, wavenumbers_original, bestHyperparams.binningFactor);
                [X_test_processed, ~] = bin_spectra(X_outer_test, wavenumbers_original, bestHyperparams.binningFactor);
            end

            selectedFeatureIndices_in_current_w = 1:size(X_train_processed, 2);
            % feature_selection_model_info = struct('method', currentPipeline.feature_selection_method); % For saving details if needed

            % Apply Feature Selection (similar to your existing logic)
            % ... (Fisher, PCA, MRMR logic as in your original script section 4) ...
            % Example for MRMR (ensure error handling is robust)
            switch lower(currentPipeline.feature_selection_method)
                case 'fisher'
                    fisherRatios = calculate_fisher_ratio(X_train_processed, y_outer_train);
                    [~, sorted_indices] = sort(fisherRatios, 'descend', 'MissingPlacement','last');
                    numFeat = min(bestHyperparams.numFisherFeatures, length(sorted_indices));
                     if numFeat == 0 && bestHyperparams.numFisherFeatures > 0 && ~isempty(sorted_indices)
                        numFeat = 1; % Select at least one if possible and requested
                        warning('Fisher selected 0 features, forcing to 1. Check Fisher scores.');
                    elseif numFeat == 0 && isempty(sorted_indices)
                         selectedFeatureIndices_in_current_w = 1:size(X_train_processed,2); % Fallback
                         fprintf('Fisher: No features to select (empty sorted_indices). Using all %d.\n', size(X_train_processed,2));
                    else
                        selectedFeatureIndices_in_current_w = sorted_indices(1:numFeat);
                    end
                    % feature_selection_model_info.num_selected = length(selectedFeatureIndices_in_current_w);
                    fprintf('      Fisher: Selected %d features for outer fold.\n', length(selectedFeatureIndices_in_current_w));
                
                case 'pca'
                    if size(X_train_processed,2) > 0 && size(X_train_processed,1) > 1 && size(X_train_processed,1) > size(X_train_processed,2)
                        [coeff_pca, score_train_pca, ~, ~, explained_pca, mu_pca] = pca(X_train_processed);
                        numComponents = 0;
                        if isfield(bestHyperparams, 'pcaVarianceToExplain')
                            cumulativeExplained = cumsum(explained_pca);
                            numComponents = find(cumulativeExplained >= bestHyperparams.pcaVarianceToExplain*100, 1, 'first');
                            if isempty(numComponents), numComponents = size(coeff_pca,2); end
                        else % numPCAComponents
                            numComponents = min(bestHyperparams.numPCAComponents, size(coeff_pca,2));
                        end
                        
                        if numComponents == 0
                           fprintf('      PCA: Resulted in 0 components. Using all %d original (binned) features.\n', size(X_train_processed,2));
                        else
                            X_train_processed = score_train_pca(:, 1:numComponents);
                            X_test_processed = (X_test_processed - mu_pca) * coeff_pca(:, 1:numComponents);
                            selectedFeatureIndices_in_current_w = 1:numComponents; % These are now PC indices
                            fprintf('      PCA: Selected %d components explaining %.2f%% variance.\n', numComponents, cumulativeExplained(numComponents));
                        end
                    else
                        fprintf('      PCA: Skipped due to data dimensions N <= P or P=0. Using all %d features.\n', size(X_train_processed,2));
                    end

                case 'mrmr'
                    y_outer_train_cat = categorical(y_outer_train); % fscmrmr needs categorical Y
                    numFeatToSelect = min(bestHyperparams.numMRMRFeatures, size(X_train_processed,2));
                    if numFeatToSelect <=0 || size(X_train_processed,2) == 0
                         selectedFeatureIndices_in_current_w = 1:size(X_train_processed,2);
                    else
                        try
                            [ranked_indices, ~] = fscmrmr(X_train_processed, y_outer_train_cat);
                            actual_num_to_take = min(numFeatToSelect, length(ranked_indices));
                            if actual_num_to_take > 0
                                selectedFeatureIndices_in_current_w = ranked_indices(1:actual_num_to_take);
                            else
                                selectedFeatureIndices_in_current_w = 1:size(X_train_processed,2); % Fallback
                            end
                        catch ME_fscmrmr_outer
                            fprintf('ERROR during fscmrmr in outer fold: %s. Using all features.\n', ME_fscmrmr_outer.message);
                            selectedFeatureIndices_in_current_w = 1:size(X_train_processed,2); 
                        end
                    end
                    fprintf('      MRMR: Selected %d features for outer fold.\n', length(selectedFeatureIndices_in_current_w));
                case 'none'
                    fprintf('      No explicit feature selection beyond binning.\n');
            end % End switch feature_selection_method

            if isempty(selectedFeatureIndices_in_current_w) && size(X_train_processed, 2) > 0
                 fprintf('    WARNING: Feature selection resulted in zero features. Using all %d (binned) features.\n', size(X_train_processed,2));
                 selectedFeatureIndices_in_current_w = 1:size(X_train_processed, 2);
            elseif isempty(X_train_processed) || size(X_train_processed,2) == 0
                outerFoldMetrics(kOuter, :) = NaN; continue; % Cannot train if no features
            end

            X_fs_train = X_train_processed(:, selectedFeatureIndices_in_current_w);
            X_fs_test  = X_test_processed(:, selectedFeatureIndices_in_current_w);
            
            trainedClassifier = [];
            if isempty(X_fs_train) || size(X_fs_train,1) < 2 || length(unique(y_outer_train)) < 2
                fprintf('      Skipping classifier training: insufficient data after FS.\n');
                outerFoldMetrics(kOuter, :) = NaN; continue;
            end

            % Train Classifier
            switch lower(currentPipeline.classifier)
                case 'lda'
                    if size(X_fs_train, 2) == 1 && var(X_fs_train) < 1e-9
                        warning('LDA training: Single feature with (near) zero variance. Skipping LDA.');
                        outerFoldMetrics(kOuter, :) = NaN; continue;
                    end
                    try
                        trainedClassifier = fitcdiscr(X_fs_train, y_outer_train);
                    catch ME_fitlda
                         fprintf('ERROR fitting LDA: %s. Skipping fold.\n', ME_fitlda.message);
                         outerFoldMetrics(kOuter, :) = NaN; continue;
                    end
            end

            % Evaluate Classifier
            if ~isempty(trainedClassifier) && ~isempty(X_fs_test) && ~isempty(y_outer_test)
                try
                    [y_pred_outer, y_scores_outer] = predict(trainedClassifier, X_fs_test);
                    positiveClassLabel = 3; 
                    classOrder = trainedClassifier.ClassNames;
                    positiveClassColIdx = find(classOrder == positiveClassLabel); % Simplified, adjust if class names can be non-numeric
                     if isempty(positiveClassColIdx) || ~(isnumeric(positiveClassColIdx)) || max(positiveClassColIdx) > size(y_scores_outer,2)
                        warning('Positive class label %d not found or scores issue. Metrics will be NaN.', positiveClassLabel);
                        outerFoldMetrics(kOuter, :) = NaN;
                    else
                        scores_for_positive_class = y_scores_outer(:, positiveClassColIdx);
                        currentFoldMetricsStruct = calculate_performance_metrics(y_outer_test, y_pred_outer, scores_for_positive_class, positiveClassLabel, metricNames);
                        outerFoldMetrics(kOuter, :) = cell2mat(struct2cell(currentFoldMetricsStruct))';
                        fprintf('      Outer fold %d test metrics: Acc=%.3f, Sens_WHO3=%.3f, F2_WHO3=%.3f\n', ...
                            kOuter, currentFoldMetricsStruct.Accuracy, currentFoldMetricsStruct.Sensitivity_WHO3, currentFoldMetricsStruct.F2_WHO3);
                    end
                catch ME_pred_eval
                    fprintf('ERROR during prediction/evaluation: %s\n', ME_pred_eval.message);
                    outerFoldMetrics(kOuter, :) = NaN;
                end
            else
                 outerFoldMetrics(kOuter, :) = NaN;
            end
        end % End of kOuter loop

        meanOuterFoldMetrics = nanmean(outerFoldMetrics, 1);
        stdOuterFoldMetrics = nanstd(outerFoldMetrics, 0, 1);

        fprintf('    --- Pipeline %s (Strategy: %s) Average Performance ---\n', currentPipeline.name, currentOutlierStrategy);
        metricsTable = array2table([meanOuterFoldMetrics', stdOuterFoldMetrics'], ...
            'VariableNames', {'Mean', 'StdDev'}, 'RowNames', metricNames');
        disp(metricsTable);

        pipelineSummary = struct();
        pipelineSummary.pipelineConfig = currentPipeline;
        pipelineSummary.outerFoldMetrics_raw = outerFoldMetrics;
        pipelineSummary.outerFoldMetrics_mean = meanOuterFoldMetrics;
        pipelineSummary.outerFoldMetrics_std = stdOuterFoldMetrics;
        pipelineSummary.outerFoldBestHyperparams = outerFoldBestHyperparams;
        % pipelineSummary.outerFoldSelectedFeaturesInfo = outerFoldSelectedFeaturesInfo; % If saving
        pipelineSummary.metricNames = metricNames;
        currentStrategyPipelinesResults{iPipeline} = pipelineSummary;
    end % End of iPipeline loop

    overallComparisonResults.(sprintf('Strategy_%s', currentOutlierStrategy)).allPipelinesResults = currentStrategyPipelinesResults;
    overallComparisonResults.(sprintf('Strategy_%s', currentOutlierStrategy)).pipelines = pipelines; % Assuming pipelines def is same
    overallComparisonResults.(sprintf('Strategy_%s', currentOutlierStrategy)).metricNames = metricNames;

    % Save results for the current strategy (as before)
    resultsFilename_strat = fullfile(resultsPath, sprintf('%s_Phase2_AllPipelineResults_Strat_%s.mat', dateStrForFilenames, currentOutlierStrategy));
    save(resultsFilename_strat, 'currentStrategyPipelinesResults', 'pipelines', 'metricNames', 'numOuterFolds', 'numInnerFolds', 'currentOutlierStrategy');
    fprintf('Phase 2 results for strategy %s saved to: %s\n', currentOutlierStrategy, resultsFilename_strat);
    
end % --- END OF MAIN LOOP FOR OUTLIER STRATEGIES ---

%% 5. Select Best Overall Pipeline FOR EACH STRATEGY (and report)
% This section remains largely the same, but operates on overallComparisonResults
fprintf('\n\n====================================================================\n');
fprintf('   SELECTING BEST PIPELINE FOR EACH STRATEGY\n');
fprintf('====================================================================\n');
bestPipelineInfoPerStrategy = struct();

for iStrategy = 1:length(outlierStrategiesToCompare)
    currentStrategyAbbrev = outlierStrategiesToCompare{iStrategy};
    currentStrategyNameFull = sprintf('Strategy_%s', currentStrategyAbbrev);
    
    if ~isfield(overallComparisonResults, currentStrategyNameFull) || ...
       ~isfield(overallComparisonResults.(currentStrategyNameFull), 'allPipelinesResults')
        fprintf('Results for strategy %s not found in overallComparisonResults. Skipping best pipeline selection for it.\n', currentStrategyAbbrev);
        continue;
    end
    
    strategyResults = overallComparisonResults.(currentStrategyNameFull);
    pipelinesForStrategy = overallComparisonResults.pipelines; % Assuming pipelines are same
    metricNamesForStrategy = overallComparisonResults.metricNames; % Assuming metric names are same

    bestF2Score_strat = -Inf;
    bestPipelineIdx_strat = -1;
    f2_idx_report = find(strcmpi(metricNamesForStrategy, 'F2_WHO3'));
    if isempty(f2_idx_report)
        error('F2_WHO3 metric not found for strategy %s.', currentStrategyAbbrev);
    end

    for iPipeline = 1:length(pipelinesForStrategy)
        if ~isempty(strategyResults.allPipelinesResults{iPipeline}) && ...
           isstruct(strategyResults.allPipelinesResults{iPipeline}) && ...
           isfield(strategyResults.allPipelinesResults{iPipeline}, 'outerFoldMetrics_mean') && ...
           length(strategyResults.allPipelinesResults{iPipeline}.outerFoldMetrics_mean) >= f2_idx_report
            
            currentMeanF2_strat = strategyResults.allPipelinesResults{iPipeline}.outerFoldMetrics_mean(f2_idx_report);
            fprintf('Strategy %s - Pipeline: %s, Mean F2_WHO3: %.4f\n', ...
                currentStrategyAbbrev, pipelinesForStrategy{iPipeline}.name, currentMeanF2_strat);
            if ~isnan(currentMeanF2_strat) && currentMeanF2_strat > bestF2Score_strat
                bestF2Score_strat = currentMeanF2_strat;
                bestPipelineIdx_strat = iPipeline;
            end
        else
            fprintf('Strategy %s - Pipeline: %s results missing or invalid.\n', ...
                currentStrategyAbbrev, pipelinesForStrategy{iPipeline}.name);
        end
    end

    if bestPipelineIdx_strat > 0
        bestPipelineSummary_strat = strategyResults.allPipelinesResults{bestPipelineIdx_strat};
        fprintf('\nBest Pipeline for Strategy %s: %s with Mean F2_WHO3 = %.4f\n', ...
            currentStrategyAbbrev, bestPipelineSummary_strat.pipelineConfig.name, bestF2Score_strat);
        bestPipelineInfoPerStrategy.(currentStrategyNameFull) = bestPipelineSummary_strat;
        
        bestModelInfoFilename_strat = fullfile(modelsPath, sprintf('%s_Phase2_BestPipelineInfo_Strat_%s.mat', dateStrForFilenames, currentStrategyAbbrev));
        save(bestModelInfoFilename_strat, 'bestPipelineSummary_strat', 'currentOutlierStrategy'); % Save with the strategy name used in loop
        fprintf('Best pipeline info for strategy %s saved to: %s\n', currentStrategyAbbrev, bestModelInfoFilename_strat);
    else
        fprintf('\nNo suitable pipeline found for strategy %s.\n', currentStrategyAbbrev);
        bestPipelineInfoPerStrategy.(currentStrategyNameFull) = []; % Store empty if none found
    end
end

%% 6. Visualization and Comparison of Strategies (NEW SECTION)
fprintf('\n\n====================================================================\n');
fprintf('   COMPARING OUTLIER STRATEGY EFFECTS ACROSS ALL PIPELINES\n');
fprintf('====================================================================\n');

% --- Prepare Data for Comparison Plot ---
pipelines_compare = overallComparisonResults.pipelines; % Defined once
metricNames_compare = overallComparisonResults.metricNames; % Defined once
numPipelines_compare = length(pipelines_compare);
pipelineNamesList_compare = cell(numPipelines_compare, 1);
for i=1:numPipelines_compare, pipelineNamesList_compare{i} = pipelines_compare{i}.name; end

targetMetricCompare = 'F2_WHO3';
targetMetricIdxCompare = find(strcmpi(metricNames_compare, targetMetricCompare));
if isempty(targetMetricIdxCompare)
    error('Target metric "%s" for comparison not found.', targetMetricCompare);
end

mean_OR_values_compare = NaN(numPipelines_compare, 1);
mean_AND_values_compare = NaN(numPipelines_compare, 1);
std_OR_values_compare = NaN(numPipelines_compare, 1); 
std_AND_values_compare = NaN(numPipelines_compare, 1);

for iPipeline = 1:numPipelines_compare
    % OR Strategy Results
    if isfield(overallComparisonResults, 'Strategy_OR') && ...
       isfield(overallComparisonResults.Strategy_OR, 'allPipelinesResults') && ...
       iPipeline <= length(overallComparisonResults.Strategy_OR.allPipelinesResults) && ...
       ~isempty(overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}) && ...
       isfield(overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}, 'outerFoldMetrics_mean')
        
        mean_metrics_or = overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}.outerFoldMetrics_mean;
        std_metrics_or  = overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}.outerFoldMetrics_std;
        if length(mean_metrics_or) >= targetMetricIdxCompare
            mean_OR_values_compare(iPipeline) = mean_metrics_or(targetMetricIdxCompare);
            std_OR_values_compare(iPipeline) = std_metrics_or(targetMetricIdxCompare);
        end
    end
    % AND Strategy Results
    if isfield(overallComparisonResults, 'Strategy_AND') && ...
       isfield(overallComparisonResults.Strategy_AND, 'allPipelinesResults') && ...
       iPipeline <= length(overallComparisonResults.Strategy_AND.allPipelinesResults) && ...
       ~isempty(overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}) && ...
       isfield(overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}, 'outerFoldMetrics_mean')

        mean_metrics_and = overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}.outerFoldMetrics_mean;
        std_metrics_and  = overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}.outerFoldMetrics_std;
        if length(mean_metrics_and) >= targetMetricIdxCompare
            mean_AND_values_compare(iPipeline) = mean_metrics_and(targetMetricIdxCompare);
            std_AND_values_compare(iPipeline) = std_metrics_and(targetMetricIdxCompare);
        end
    end
end

% --- Bar Chart Comparison ---
figCompareBar = figure('Name', ['Comparison of Outlier Strategies - Mean ' targetMetricCompare], 'Position', [100, 100, 1200, 700]);
bar_data_compare = [mean_OR_values_compare, mean_AND_values_compare];
b_compare = bar(bar_data_compare);
hold on;
numGroups = size(bar_data_compare, 1);
numBarsPerGroup = size(bar_data_compare, 2); % Should be 2 (OR and AND)
groupWidth = min(0.8, numBarsPerGroup/(numBarsPerGroup + 1.5));
for iBar = 1:numBarsPerGroup
    x_bar_centers = (1:numGroups) - groupWidth/2 + (2*iBar-1) * groupWidth / (2*numBarsPerGroup);
    if iBar == 1 % OR strategy
        errorbar(x_bar_centers, mean_OR_values_compare, std_OR_values_compare, 'k.', 'HandleVisibility','off');
    else % AND strategy
        errorbar(x_bar_centers, mean_AND_values_compare, std_AND_values_compare, 'k.', 'HandleVisibility','off');
    end
end
hold off;

b_compare(1).FaceColor = [0.9, 0.6, 0.4]; % Orange for OR
b_compare(2).FaceColor = [0.4, 0.702, 0.902]; % Blue for AND

xticks(1:numPipelines_compare);
xticklabels(pipelineNamesList_compare);
xtickangle(45);
ylabel(['Mean ' strrep(targetMetricCompare, '_', ' ')]);
title({['Comparison of Outlier Removal Strategies on Mean ' strrep(targetMetricCompare, '_', ' ')], ...
       '(Error bars represent +/-1 Std. Dev. of outer CV fold scores)'}, 'FontWeight', 'normal');
legend({'T2 OR Q Strategy', 'T2 AND Q Strategy (Consensus)'}, 'Location', 'NorthEastOutside');
grid on;
valid_bar_data = bar_data_compare(~isnan(bar_data_compare));
valid_std_data = [std_OR_values_compare(~isnan(std_OR_values_compare)); std_AND_values_compare(~isnan(std_AND_values_compare))];
if ~isempty(valid_bar_data)
    max_val_for_ylim = max(valid_bar_data(:) + valid_std_data(:), [], 'omitnan');
    if isempty(max_val_for_ylim) || isnan(max_val_for_ylim) || max_val_for_ylim == 0, max_val_for_ylim = 0.1; end
     min_val_for_ylim = min(valid_bar_data(:) - valid_std_data(:), [], 'omitnan');
    if isempty(min_val_for_ylim) || isnan(min_val_for_ylim), min_val_for_ylim = 0; end
    ylim_padding = (max_val_for_ylim - min_val_for_ylim) * 0.1;
    if ylim_padding == 0, ylim_padding = 0.05; end
    final_ylim = [max(0, min_val_for_ylim - ylim_padding), max_val_for_ylim + ylim_padding];
    if final_ylim(1) >= final_ylim(2), final_ylim = [0, final_ylim(2)+0.1]; end % Ensure valid range
    ylim(final_ylim);
else
    ylim([0 0.1]); % Default if no valid data
end


barPlotCompFilenameBase = fullfile(comparisonFiguresPath, sprintf('%s_BarPlot_OutlierStrategyComparison_%s', dateStrForFilenames, targetMetricCompare));
savefig(figCompareBar, [barPlotCompFilenameBase, '.fig']);
exportgraphics(figCompareBar, [barPlotCompFilenameBase, '.tiff'], 'Resolution', 300);
fprintf('Comprehensive comparison bar plot saved to: %s.(fig/tiff)\n', barPlotCompFilenameBase);

% --- Create and Save Detailed Comparison Table (CSV) ---
varNamesForDetailedTable = {'PipelineName'};
for mIdx = 1:length(metricNames_compare)
    varNamesForDetailedTable{end+1} = [metricNames_compare{mIdx} '_Mean_OR'];
    varNamesForDetailedTable{end+1} = [metricNames_compare{mIdx} '_Std_OR'];
end
for mIdx = 1:length(metricNames_compare)
    varNamesForDetailedTable{end+1} = [metricNames_compare{mIdx} '_Mean_AND'];
    varNamesForDetailedTable{end+1} = [metricNames_compare{mIdx} '_Std_AND'];
end

dataForDetailedTable = cell(numPipelines_compare, length(varNamesForDetailedTable));
for iPipeline = 1:numPipelines_compare
    dataForDetailedTable{iPipeline, 1} = pipelineNamesList_compare{iPipeline};
    idxOffset = 1; % Start after PipelineName
    
    % OR Metrics
    if isfield(overallComparisonResults, 'Strategy_OR') && iPipeline <= length(overallComparisonResults.Strategy_OR.allPipelinesResults) && ~isempty(overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline})
        results_OR_pipe = overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline};
        for mIdx = 1:length(metricNames_compare)
            dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 1} = results_OR_pipe.outerFoldMetrics_mean(mIdx);
            dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 2} = results_OR_pipe.outerFoldMetrics_std(mIdx);
        end
    else % Fill with NaNs if results are missing for this pipeline under OR
         for mIdx = 1:length(metricNames_compare)
            dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 1} = NaN;
            dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 2} = NaN;
         end
    end
    
    idxOffset = 1 + 2*length(metricNames_compare); % Move to where AND metrics start
    % AND Metrics
    if isfield(overallComparisonResults, 'Strategy_AND') && iPipeline <= length(overallComparisonResults.Strategy_AND.allPipelinesResults) && ~isempty(overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline})
        results_AND_pipe = overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline};
        for mIdx = 1:length(metricNames_compare)
            dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 1} = results_AND_pipe.outerFoldMetrics_mean(mIdx);
            dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 2} = results_AND_pipe.outerFoldMetrics_std(mIdx);
        end
    else % Fill with NaNs
         for mIdx = 1:length(metricNames_compare)
            dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 1} = NaN;
            dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 2} = NaN;
         end
    end
end
detailedComparisonCSVTable = cell2table(dataForDetailedTable, 'VariableNames', varNamesForDetailedTable);
disp('Comprehensive Comparison Table (All Metrics):');
disp(detailedComparisonCSVTable);
comparisonTableFilename = fullfile(comparisonResultsPath, sprintf('%s_OutlierStrategies_AllMetricsComparison.csv', dateStrForFilenames));
writetable(detailedComparisonCSVTable, comparisonTableFilename);
fprintf('Comprehensive comparison CSV table saved to: %s\n', comparisonTableFilename);

% --- Spider Plot Comparison (Optional, if spider_plot_R2019b.m is available) ---
% (Spider plot logic from previous response, using overallComparisonResults, can be placed here)
% (Ensure paths and variable names are consistent with this script's structure)
if exist('spider_plot_R2019b', 'file') 
    fprintf('\nAttempting to generate Spider Plot for comparison...\n');
    % ... (Code from previous response, adapted to use overallComparisonResults,
    %      pipelines_compare, metricNames_compare, etc.
    %      Save to comparisonFiguresPath)
    % Example for one pipeline:
    if numPipelines_compare > 0 && ~isempty(bestPipelineInfoPerStrategy.Strategy_OR) && ~isempty(bestPipelineInfoPerStrategy.Strategy_AND)
        % Let's plot the best OR pipeline vs best AND pipeline if they are different
        % Or just the first pipeline for an example
        idxSpiderPipe = 1; % Example: first pipeline
        
        spider_P_matrix = [];
        spider_axes_labels = {};
        metrics_for_spider = {'F2_WHO3', 'Sensitivity_WHO3', 'Specificity_WHO1', 'AUC', 'Accuracy'};

        for iMet = 1:length(metrics_for_spider)
            metric_name_spider = metrics_for_spider{iMet};
            metric_idx_spider = find(strcmpi(metricNames_compare, metric_name_spider));
            if isempty(metric_idx_spider), continue; end

            val_or = NaN; val_and = NaN;
             if isfield(overallComparisonResults.Strategy_OR.allPipelinesResults{idxSpiderPipe}, 'outerFoldMetrics_mean')
                val_or = overallComparisonResults.Strategy_OR.allPipelinesResults{idxSpiderPipe}.outerFoldMetrics_mean(metric_idx_spider);
             end
             if isfield(overallComparisonResults.Strategy_AND.allPipelinesResults{idxSpiderPipe}, 'outerFoldMetrics_mean')
                val_and = overallComparisonResults.Strategy_AND.allPipelinesResults{idxSpiderPipe}.outerFoldMetrics_mean(metric_idx_spider);
             end
            spider_P_matrix = [spider_P_matrix, [val_or; val_and]];
            spider_axes_labels{end+1} = strrep(metric_name_spider, '_', ' ');
        end
        
        if ~isempty(spider_P_matrix)
            spider_P_matrix(isnan(spider_P_matrix)) = 0; % Handle NaNs for spider plot
            figure('Name', sprintf('Spider Plot Comparison for %s', pipelines_compare{idxSpiderPipe}.name));
            spider_plot_R2019b(spider_P_matrix', ... % Transpose P so rows are strategies
                'AxesLabels', spider_axes_labels, ...
                'AxesLimits', repmat([0;1], 1, length(spider_axes_labels)), ...
                'FillOption', 'on', ...
                'FillTransparency', [0.2, 0.1], ...
                'Color', [[0.9,0.6,0.4]; [0.4,0.702,0.902]], ...
                'LineWidth', 1.5);
            title(sprintf('Comparison for %s: OR vs AND Strategy', pipelines_compare{idxSpiderPipe}.name));
            legend({'T2 OR Q', 'T2 AND Q (Consensus)'}, 'Location', 'bestoutside');
            
            spiderPlotFilename = fullfile(comparisonFiguresPath, sprintf('%s_SpiderPlot_CompareStrategies_%s.tiff', dateStrForFilenames, pipelines_compare{idxSpiderPipe}.name));
            exportgraphics(gcf, spiderPlotFilename, 'Resolution', 300);
            savefig(gcf, strrep(spiderPlotFilename, '.tiff','.fig'));
            fprintf('Spider plot for %s saved to %s\n', pipelines_compare{idxSpiderPipe}.name, spiderPlotFilename);
        else
            fprintf('Not enough data to generate spider plot for pipeline %s.\n', pipelines_compare{idxSpiderPipe}.name);
        end
    end
else
    fprintf('spider_plot_R2019b.m not found. Skipping spider plot.\n');
end


fprintf('\nPHASE 2 Processing & Outlier Strategy Comparison Complete: %s\n', string(datetime('now')));