function run_phase2_model_selection_comparative(cfg)
%RUN_PHASE2_MODEL_SELECTION_COMPARATIVE
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

if nargin < 1
    cfg = struct();
end
if ~isfield(cfg, 'projectRoot')
    cfg.projectRoot = pwd;
end
if ~isfield(cfg, 'outlierStrategiesToCompare')
    cfg.outlierStrategiesToCompare = {'OR', 'AND'};
end

% --- Define Paths ---
P = setup_project_paths(cfg.projectRoot, 'Phase2');
dataPath    = P.dataPath;
resultsPath = P.resultsPath;
modelsPath  = P.modelsPath;
figuresPath = P.figuresPath;

comparisonFiguresPath = fullfile(P.projectRoot, 'figures', 'OutlierStrategyComparison'); % For new comparison plots
comparisonResultsPath = fullfile(P.projectRoot, 'results', 'OutlierStrategyComparison'); % For new comparison tables/data

dirToEnsure = {resultsPath, modelsPath, figuresPath, comparisonFiguresPath, comparisonResultsPath};
for i=1:length(dirToEnsure), if ~isfolder(dirToEnsure{i}), mkdir(dirToEnsure{i}); end, end

dateStrForFilenames = string(datetime('now','Format','yyyyMMdd'));

% --- Define Outlier Strategies to Compare & Result Storage ---
outlierStrategiesToCompare = cfg.outlierStrategiesToCompare;
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
        p.hyperparameters_to_tune = {'binningFactor', 'fisherFeaturePercent'};
        p.binningFactors = [1, 2, 4, 8, 16];
        p.fisherFeaturePercent_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5];
        pipelineIdx = pipelineIdx + 1; pipelines{pipelineIdx} = p;
        % --- Pipeline 3: PCA + LDA ---
        p = struct(); p.name = 'PCALDA'; p.feature_selection_method = 'pca'; p.classifier = 'LDA';
        p.hyperparameters_to_tune = {'binningFactor', 'pcaVarianceToExplain'};
        p.binningFactors = [1, 2, 4, 8, 16]; p.pcaVarianceToExplain_range = [0.90, 0.95, 0.99];
        pipelineIdx = pipelineIdx + 1; pipelines{pipelineIdx} = p;
        % --- Pipeline 4: MRMR + LDA ---
        p = struct();
        p.name = 'MRMRLDA';
        p.feature_selection_method = 'mrmr';
        p.classifier = 'LDA';
        p.hyperparameters_to_tune = {'binningFactor', 'mrmrFeaturePercent'};
        p.binningFactors = [1, 2, 4, 8, 16];
        p.mrmrFeaturePercent_range = [0.05, 0.1, 0.2, 0.3, 0.4];
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
                    numFeat = ceil(bestHyperparams.fisherFeaturePercent * length(sorted_indices));
                    numFeat = min(numFeat, length(sorted_indices));
                     if numFeat == 0 && bestHyperparams.fisherFeaturePercent > 0 && ~isempty(sorted_indices)
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
                    numFeatToSelect = ceil(bestHyperparams.mrmrFeaturePercent * size(X_train_processed,2));
                    numFeatToSelect = min(numFeatToSelect, size(X_train_processed,2));
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
        %% Train final model on full training data using aggregated best hyperparameters
        aggHyper = aggregate_best_hyperparams(outerFoldBestHyperparams);
        try
            [finalModel, selectedIdx, selectedWn] = train_final_pipeline_model(...
                X_train_full, y_train_full, wavenumbers_original, currentPipeline, aggHyper);
            modelFilename = fullfile(modelsPath, sprintf('%s_Phase2_%s_Model_Strat_%s.mat', ...
                dateStrForFilenames, currentPipeline.name, currentOutlierStrategy));
            save(modelFilename, 'finalModel', 'aggHyper', 'selectedIdx', 'selectedWn');
            pipelineSummary.finalModelFile = modelFilename;
            fprintf('Saved final model for %s (strategy %s) to %s\n', currentPipeline.name, currentOutlierStrategy, modelFilename);
        catch ME_train
            warning('Failed to train final model for %s: %s', currentPipeline.name, ME_train.message);
            pipelineSummary.finalModelFile = '';
        end
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

%% 6. Save Overall Comparison Data for Visualization Script (NEW SECTION)
% =========================================================================
fprintf('\n--- Saving Overall Comparison Data for Visualization Script ---\n');

% These paths and date string should be defined earlier in your script.
% Ensure they are accessible here.
% projectRoot = pwd; % Defined at the start of your script
% comparisonResultsPath = fullfile(projectRoot, 'results', 'OutlierStrategyComparison_Results'); % Defined at the start
% dateStrForFilenames = string(datetime('now','Format','yyyyMMdd')); % Defined at the start

if exist('overallComparisonResults', 'var') && isstruct(overallComparisonResults) && ~isempty(fieldnames(overallComparisonResults))
    % Define the filename for the overall comparison data
    % This filename pattern matches what 'run_visualize_project_results.m' is looking for.
    overallComparisonDataFilename = fullfile(comparisonResultsPath, sprintf('%s_Phase2_OverallComparisonData.mat', dateStrForFilenames));
    
    % Save the overallComparisonResults structure.
    % The 'run_visualize_project_results.m' script specifically loads a variable named 'overallComparisonResults'
    % from this .mat file.
    save(overallComparisonDataFilename, 'overallComparisonResults', '-v7.3');
    
    fprintf('Overall Phase 2 comparison data saved to: %s\n', overallComparisonDataFilename);
else
    fprintf('Warning: The "overallComparisonResults" variable was not found or is empty. \n');
    fprintf('The overall comparison .mat file required by the visualization script was NOT saved.\n');
    fprintf('Please ensure "overallComparisonResults" is correctly populated before this section.\n');
end

% This should be one of the last outputs of your script.
fprintf('\nPHASE 2 Processing & Outlier Strategy Comparison Complete (with OverallComparisonData save): %s\n', string(datetime('now')));
end

function aggHyper = aggregate_best_hyperparams(hyperparamCell)
    aggHyper = struct();
    if isempty(hyperparamCell)
        return;
    end

    % Filter to only non-empty struct entries to avoid errors with fieldnames
    isValidStruct = cellfun(@(c) isstruct(c) && ~isempty(c), hyperparamCell);
    hyperparamCell = hyperparamCell(isValidStruct);
    if isempty(hyperparamCell)
        return;
    end

    allFieldsNested = cellfun(@fieldnames, hyperparamCell, 'UniformOutput', false);
    allFields = unique([allFieldsNested{:}]);

    for f = 1:numel(allFields)
        fname = allFields{f};
        values = []; %#ok<AGROW>
        for k = 1:numel(hyperparamCell)
            hp = hyperparamCell{k};
            if isfield(hp, fname)
                values(end+1) = hp.(fname); %#ok<AGROW>
            end
        end
        if ~isempty(values)
            aggHyper.(fname) = mode(values);
        end
    end
end

function [modelStruct, selectedIdx, selectedWn] = train_final_pipeline_model(X, y, wn, pipelineConfig, hp)
    modelStruct = struct();
    selectedIdx = 1:size(X,2);
    selectedWn = wn;
    Xp = X;
    currentWn = wn;
    if isfield(hp,'binningFactor') && hp.binningFactor > 1
        [Xp,currentWn] = bin_spectra(X, wn, hp.binningFactor);
    end
    switch lower(pipelineConfig.feature_selection_method)
        case 'fisher'
            fr = calculate_fisher_ratio(Xp, y);
            [~, order] = sort(fr,'descend','MissingPlacement','last');
            numF = ceil(hp.fisherFeaturePercent * numel(order));
            numF = max(1, min(numF, numel(order)));
            selectedIdx = order(1:numF);
            Xp = Xp(:,selectedIdx);
            selectedWn = currentWn(selectedIdx);
        case 'pca'
            [coeff, score, ~, ~, explained, mu] = pca(Xp);
            numC = find(cumsum(explained) >= hp.pcaVarianceToExplain*100,1,'first');
            if isempty(numC)
                numC = size(coeff,2);
            end
            selectedIdx = 1:numC;
            Xp = score(:,selectedIdx);
            modelStruct.PCACoeff = coeff(:,selectedIdx);
            modelStruct.PCAMu = mu;
            selectedWn = selectedIdx;
        case 'mrmr'
            numF = ceil(hp.mrmrFeaturePercent * size(Xp,2));
            numF = max(1, min(numF, size(Xp,2)));
            [rankedIdx,~] = fscmrmr(Xp,categorical(y));
            selectedIdx = rankedIdx(1:numF);
            Xp = Xp(:,selectedIdx);
            selectedWn = currentWn(selectedIdx);
        otherwise
            selectedIdx = 1:size(Xp,2);
            selectedWn = currentWn;
    end
    lda = fitcdiscr(Xp, y);
    modelStruct.LDAModel = lda;
if isfield(hp, 'binningFactor') && ~isempty(hp.binningFactor)
    modelStruct.binningFactor = hp.binningFactor;
else
    modelStruct.binningFactor = 1; % Assign default value of 1 if not present
end
    modelStruct.featureSelectionMethod = pipelineConfig.feature_selection_method;
    modelStruct.selectedFeatureIndices = selectedIdx;
    modelStruct.selectedWavenumbers = selectedWn;
    modelStruct.originalWavenumbers = wn;
    modelStruct.binnedWavenumbers = currentWn;
end
