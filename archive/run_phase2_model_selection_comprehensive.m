% run_phase2_model_selection_comprehensive.m
%
% Main script for Phase 2: Model and Feature Selection Pipelines.
% Implements nested cross-validation to find the best combination of
% preprocessing, feature selection, and classifier.
% THIS VERSION COMPARES TWO OUTLIER REMOVAL STRATEGIES ('OR' and 'AND')
% IN A SINGLE RUN.
%
% Focus: Prioritize correct identification of WHO-3.
% Method: Probe-wise 5-fold nested cross-validation.
%
% Date: 2025-05-18 (Modified for comprehensive strategy comparison)

%% 0. Initialization
fprintf('PHASE 2: Model and Feature Selection (Comprehensive Outlier Strategy Comparison) - %s\n', string(datetime('now')));

% --- Define Paths (Simplified) ---
projectRoot = pwd; % Assumes current working directory IS the project root.
if ~exist(fullfile(projectRoot, 'src'), 'dir') || ~exist(fullfile(projectRoot, 'data'), 'dir')
    error(['Project structure not found. Please ensure MATLAB''s "Current Folder" is set to your ' ...
           'main project root directory before running. Current directory is: %s'], projectRoot);
end

srcPath       = fullfile(projectRoot, 'src');
helperFunPath = fullfile(srcPath, 'helper_functions');
if ~exist(helperFunPath, 'dir')
    error('The ''helper_functions'' directory was not found inside ''%s''.', srcPath);
end
if ~contains(path, helperFunPath)
    addpath(helperFunPath);
end

dataPath      = fullfile(projectRoot, 'data');
resultsPath   = fullfile(projectRoot, 'results', 'Phase2'); % Specific to Phase 2
modelsPath    = fullfile(projectRoot, 'models', 'Phase2');   % Specific to Phase 2
figuresPath   = fullfile(projectRoot, 'figures', 'Phase2'); % Specific to Phase 2
% For new comparison figures
comparisonFiguresPath = fullfile(projectRoot, 'figures', 'OutlierStrategyComparison');
if ~exist(comparisonFiguresPath, 'dir'), mkdir(comparisonFiguresPath); end


if ~exist(resultsPath, 'dir'), mkdir(resultsPath); end
if ~exist(modelsPath, 'dir'), mkdir(modelsPath); end
if ~exist(figuresPath, 'dir'), mkdir(figuresPath); end

dateStrForFilenames = string(datetime('now','Format','yyyyMMdd'));

% +++ DEFINE OUTLIER STRATEGIES TO COMPARE +++
outlierStrategiesToCompare = {'OR', 'AND'}; % Will loop through these
overallComparisonResults = struct(); % To store results for each strategy

%% --- MAIN LOOP FOR OUTLIER STRATEGIES ---
for iStrategy = 1:length(outlierStrategiesToCompare)
    currentOutlierStrategy = outlierStrategiesToCompare{iStrategy};
    fprintf('\n\n====================================================================\n');
    fprintf('   PROCESSING WITH OUTLIER STRATEGY: T2 %s Q\n', currentOutlierStrategy);
    fprintf('====================================================================\n');

    %% 1. Load Data (Specific to currentOutlierStrategy)
    fprintf('\n--- 1. Loading Data (Outlier Strategy: %s) ---\n', currentOutlierStrategy);

    if strcmpi(currentOutlierStrategy, 'OR')
        inputDataFilePattern = '*_training_set_no_outliers_T2orQ.mat';
    elseif strcmpi(currentOutlierStrategy, 'AND')
        inputDataFilePattern = '*_training_set_no_outliers_T2andQ.mat';
    else
        error('Invalid outlierStrategy specified in loop. Choose "OR" or "AND".');
    end

    cleanedDataFiles = dir(fullfile(dataPath, inputDataFilePattern));
    if isempty(cleanedDataFiles)
        error('No cleaned training set file found for strategy "%s" in %s matching pattern %s. Ensure the outlier removal script for this strategy has been run and saved files to %s.', ...
              currentOutlierStrategy, dataPath, inputDataFilePattern, dataPath);
    end
    [~,idxSortCleaned] = sort([cleanedDataFiles.datenum],'descend');
    inputDataFile = fullfile(dataPath, cleanedDataFiles(idxSortCleaned(1)).name);
    fprintf('Loading cleaned training data (Strategy: %s) from: %s\n', currentOutlierStrategy, inputDataFile);

    try
        if strcmpi(currentOutlierStrategy, 'OR')
            loadedData = load(inputDataFile, ...
                               'X_train_no_outliers_OR', 'y_train_no_outliers_OR_num', ...
                               'Patient_ID_train_no_outliers_OR', 'wavenumbers_roi');
            X_train_full = loadedData.X_train_no_outliers_OR;
            y_train_full = loadedData.y_train_no_outliers_OR_num;
            probeIDs_train_full = loadedData.Patient_ID_train_no_outliers_OR;
        else % AND strategy
            loadedData = load(inputDataFile, ...
                               'X_train_no_outliers_AND', 'y_train_no_outliers_AND_num', ...
                               'Patient_ID_train_no_outliers_AND', 'wavenumbers_roi');
            X_train_full = loadedData.X_train_no_outliers_AND;
            y_train_full = loadedData.y_train_no_outliers_AND_num;
            probeIDs_train_full = loadedData.Patient_ID_train_no_outliers_AND;
        end
        
        if isfield(loadedData, 'wavenumbers_roi')
            wavenumbers_original = loadedData.wavenumbers_roi;
        else % Fallback if not saved with training sets (should be there based on run_outlier_detection_pca2.m)
            wavenumbers_data_fallback = load(fullfile(dataPath, 'wavenumbers.mat'), 'wavenumbers_roi');
            wavenumbers_original = wavenumbers_data_fallback.wavenumbers_roi;
            fprintf('Loaded wavenumbers_roi from separate file for strategy %s.\n', currentOutlierStrategy);
        end
        if iscolumn(wavenumbers_original), wavenumbers_original = wavenumbers_original'; end
        
        if isempty(X_train_full) || isempty(y_train_full) || isempty(probeIDs_train_full)
            error('One or more required datasets (X_train_full, y_train_full, probeIDs_train_full) are empty after loading for strategy %s.', currentOutlierStrategy);
        end
        if ~isnumeric(y_train_full)
             error('y_train_full is not numeric after loading for strategy %s. Type is: %s.', currentOutlierStrategy, class(y_train_full));
        end
        fprintf('Data for strategy %s loaded: %d spectra, %d features. %d unique probe IDs.\n', ...
            currentOutlierStrategy, size(X_train_full, 1), size(X_train_full, 2), length(unique(probeIDs_train_full)));

    catch ME
        fprintf('ERROR loading data for strategy %s from %s: %s\n', currentOutlierStrategy, inputDataFile, ME.message);
        rethrow(ME);
    end

    %% 2. Define Cross-Validation Parameters
    % These are generally independent of the strategy, but defined inside loop for clarity
    % if they needed to change per strategy (not the case here).
    numOuterFolds = 5;
    rng('default'); % Reset RNG for each strategy to ensure CV splits are comparable if data is same size
                    % If data sizes differ significantly, splits will inherently differ.

    [uniqueProbes, ~, groupIdxPerSpectrum] = unique(probeIDs_train_full, 'stable');
    probe_WHO_Grade = zeros(length(uniqueProbes), 1);
    for i = 1:length(uniqueProbes)
        probeSpectraLabels = y_train_full(groupIdxPerSpectrum == i);
        if any(probeSpectraLabels == 3) % Prioritize WHO-3 if mixed labels in a probe
            probe_WHO_Grade(i) = 3;
        else
            probe_WHO_Grade(i) = mode(probeSpectraLabels); % Majority vote for others
        end
    end
    
    % Check if there are enough samples in each class at the probe level for cvpartition
    classCountsProbeLevel = histcounts(categorical(probe_WHO_Grade));
    if any(classCountsProbeLevel < numOuterFolds) && length(uniqueProbes) >= numOuterFolds
        warning('Strategy %s: Not enough probes in at least one class (%s) for stratified %d-fold CV. CV may be unstratified or error.', ...
            currentOutlierStrategy, mat2str(classCountsProbeLevel), numOuterFolds);
        % cvpartition might handle this by itself or error. If it errors, may need to use unstratified or reduce folds.
        try
            outerCV_probeLevel = cvpartition(probe_WHO_Grade, 'KFold', numOuterFolds);
        catch ME_cv_strat
            fprintf('Stratified CV failed for strategy %s. Attempting unstratified.\n', currentOutlierStrategy);
            try
                outerCV_probeLevel = cvpartition(length(uniqueProbes), 'KFold', numOuterFolds); % Unstratified
            catch ME_cv_unstrat
                 error('Could not create outer CV partition for strategy %s even with unstratified. Error: %s', currentOutlierStrategy, ME_cv_unstrat.message);
            end
        end
    elseif length(uniqueProbes) < numOuterFolds
         error('Strategy %s: Number of unique probes (%d) is less than numOuterFolds (%d). Cannot perform CV.', currentOutlierStrategy, length(uniqueProbes), numOuterFolds);
    else
        outerCV_probeLevel = cvpartition(probe_WHO_Grade, 'KFold', numOuterFolds);
    end


    numInnerFolds = 3; % Can be adjusted
    metricNames = {'Accuracy', 'Sensitivity_WHO3', 'Specificity_WHO1', 'PPV_WHO3', 'NPV_WHO1', 'F1_WHO3', 'F2_WHO3', 'AUC'};

    %% 3. Define Pipelines to Evaluate
    % This definition can be outside the loop if pipelines are the same for all strategies
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
fprintf('\n\n====================================================================\n');
fprintf('   SELECTING BEST PIPELINE FOR EACH STRATEGY\n');
fprintf('====================================================================\n');

bestPipelineInfoPerStrategy = struct();

for iStrategy = 1:length(outlierStrategiesToCompare)
    currentStrategyNameFull = sprintf('Strategy_%s', outlierStrategiesToCompare{iStrategy});
    strategyResults = overallComparisonResults.(currentStrategyNameFull);
    
    bestF2Score_strat = -Inf;
    bestPipelineIdx_strat = -1;
    f2_idx_report = find(strcmpi(strategyResults.metricNames, 'F2_WHO3'));
    if isempty(f2_idx_report)
        error('F2_WHO3 metric not found in metricNames for strategy %s.', outlierStrategiesToCompare{iStrategy});
    end

    for iPipeline = 1:length(strategyResults.pipelines)
        if ~isempty(strategyResults.allPipelinesResults{iPipeline}) && ...
           isstruct(strategyResults.allPipelinesResults{iPipeline}) && ...
           isfield(strategyResults.allPipelinesResults{iPipeline}, 'outerFoldMetrics_mean') && ...
           length(strategyResults.allPipelinesResults{iPipeline}.outerFoldMetrics_mean) >= f2_idx_report
            
            currentMeanF2_strat = strategyResults.allPipelinesResults{iPipeline}.outerFoldMetrics_mean(f2_idx_report);
            fprintf('Strategy %s - Pipeline: %s, Mean F2_WHO3: %.4f\n', ...
                outlierStrategiesToCompare{iStrategy}, strategyResults.pipelines{iPipeline}.name, currentMeanF2_strat);
            if ~isnan(currentMeanF2_strat) && currentMeanF2_strat > bestF2Score_strat
                bestF2Score_strat = currentMeanF2_strat;
                bestPipelineIdx_strat = iPipeline;
            end
        else
            fprintf('Strategy %s - Pipeline: %s results missing or invalid for selection.\n', ...
                outlierStrategiesToCompare{iStrategy}, strategyResults.pipelines{iPipeline}.name);
        end
    end

    if bestPipelineIdx_strat > 0
        bestPipelineSummary_strat = strategyResults.allPipelinesResults{bestPipelineIdx_strat};
        fprintf('\nBest Pipeline for Strategy %s: %s with Mean F2_WHO3 = %.4f\n', ...
            outlierStrategiesToCompare{iStrategy}, bestPipelineSummary_strat.pipelineConfig.name, bestF2Score_strat);
        bestPipelineInfoPerStrategy.(currentStrategyNameFull) = bestPipelineSummary_strat;
        
        % Save best model info for this strategy (as before)
        bestModelInfoFilename_strat = fullfile(modelsPath, sprintf('%s_Phase2_BestPipelineInfo_Strat_%s.mat', dateStrForFilenames, outlierStrategiesToCompare{iStrategy}));
        save(bestModelInfoFilename_strat, 'bestPipelineSummary_strat', 'currentOutlierStrategy');
        fprintf('Best pipeline info for strategy %s saved to: %s\n', outlierStrategiesToCompare{iStrategy}, bestModelInfoFilename_strat);
    else
        fprintf('\nNo suitable pipeline found for strategy %s.\n', outlierStrategiesToCompare{iStrategy});
        bestPipelineInfoPerStrategy.(currentStrategyNameFull) = [];
    end
end

%% 6. Visualization and Comparison of Strategies
fprintf('\n\n====================================================================\n');
fprintf('   COMPARING OUTLIER STRATEGY EFFECTS ACROSS ALL PIPELINES\n');
fprintf('====================================================================\n');

% --- Prepare Data for Comparison Plot ---
numPipelines = length(overallComparisonResults.Strategy_OR.pipelines); % Assuming same pipelines for both
pipelineNamesList = cell(numPipelines, 1);
for i=1:numPipelines, pipelineNamesList{i} = overallComparisonResults.Strategy_OR.pipelines{i}.name; end

targetMetricCompare = 'F2_WHO3';
targetMetricIdxCompare = find(strcmpi(overallComparisonResults.Strategy_OR.metricNames, targetMetricCompare));
if isempty(targetMetricIdxCompare)
    error('Target metric "%s" for comparison not found.', targetMetricCompare);
end

mean_OR_values_compare = NaN(numPipelines, 1);
mean_AND_values_compare = NaN(numPipelines, 1);
std_OR_values_compare = NaN(numPipelines, 1); % For error bars
std_AND_values_compare = NaN(numPipelines, 1); % For error bars

for iPipeline = 1:numPipelines
    % OR Strategy
    if isfield(overallComparisonResults, 'Strategy_OR') && ...
       ~isempty(overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}) && ...
       isfield(overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}, 'outerFoldMetrics_mean')
        mean_OR_values_compare(iPipeline) = overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}.outerFoldMetrics_mean(targetMetricIdxCompare);
        std_OR_values_compare(iPipeline) = overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}.outerFoldMetrics_std(targetMetricIdxCompare);
    end
    % AND Strategy
    if isfield(overallComparisonResults, 'Strategy_AND') && ...
       ~isempty(overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}) && ...
       isfield(overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}, 'outerFoldMetrics_mean')
        mean_AND_values_compare(iPipeline) = overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}.outerFoldMetrics_mean(targetMetricIdxCompare);
        std_AND_values_compare(iPipeline) = overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}.outerFoldMetrics_std(targetMetricIdxCompare);
    end
end

% --- Bar Chart Comparison ---
figCompareBar = figure('Name', ['Comparison of Outlier Strategies - Mean ' targetMetricCompare], 'Position', [100, 100, 1200, 700]);
bar_data_compare = [mean_OR_values_compare, mean_AND_values_compare];
b_compare = bar(bar_data_compare);
hold on;

% Add error bars
numGroups = size(bar_data_compare, 1);
numBars = size(bar_data_compare, 2);
groupWidth = min(0.8, numBars/(numBars + 1.5));
for i = 1:numBars
    x_bar_centers = (1:numGroups) - groupWidth/2 + (2*i-1) * groupWidth / (2*numBars);
    if i == 1 % OR strategy
        errorbar(x_bar_centers, mean_OR_values_compare, std_OR_values_compare, 'k.', 'HandleVisibility','off');
    else % AND strategy
        errorbar(x_bar_centers, mean_AND_values_compare, std_AND_values_compare, 'k.', 'HandleVisibility','off');
    end
end
hold off;

b_compare(1).FaceColor = [0.9, 0.6, 0.4]; % Orange for OR
b_compare(2).FaceColor = [0.4, 0.702, 0.902]; % Blue for AND

xticks(1:numPipelines);
xticklabels(pipelineNamesList);
xtickangle(45);
ylabel(['Mean ' strrep(targetMetricCompare, '_', ' ')]);
title({['Comparison of Outlier Removal Strategies on Mean ' strrep(targetMetricCompare, '_', ' ')], ...
       '(Error bars represent ±1 Std. Dev. of outer CV fold scores)'});
legend({'T2 OR Q Strategy', 'T2 AND Q Strategy (Consensus)'}, 'Location', 'NorthEastOutside');
grid on;
ylim_upper_max = max(bar_data_compare(:) + [std_OR_values_compare(:); std_AND_values_compare(:)], [], 'omitnan');
if isempty(ylim_upper_max) || isnan(ylim_upper_max), ylim_upper_max = 0.1; end
ylim([0 max(ylim_upper_max, 0.1) * 1.1]);

% Save the comparison bar plot
barPlotCompFilenameBase = fullfile(comparisonFiguresPath, sprintf('%s_BarPlot_OutlierStrategyComparison_%s', dateStrForFilenames, targetMetricCompare));
savefig(figCompareBar, [barPlotCompFilenameBase, '.fig']);
exportgraphics(figCompareBar, [barPlotCompFilenameBase, '.tiff'], 'Resolution', 300);
fprintf('Comprehensive comparison bar plot saved to: %s.(fig/tiff)\n', barPlotCompFilenameBase);

% --- Create and Save Detailed Comparison Table (CSV) ---
varNamesForDetailedTable = {'PipelineName'};
metricNamesFull = overallComparisonResults.Strategy_OR.metricNames; % Assuming same for AND
for mIdx = 1:length(metricNamesFull)
    varNamesForDetailedTable{end+1} = [metricNamesFull{mIdx} '_Mean_OR'];
    varNamesForDetailedTable{end+1} = [metricNamesFull{mIdx} '_Std_OR'];
end
for mIdx = 1:length(metricNamesFull)
    varNamesForDetailedTable{end+1} = [metricNamesFull{mIdx} '_Mean_AND'];
    varNamesForDetailedTable{end+1} = [metricNamesFull{mIdx} '_Std_AND'];
end

dataForDetailedTable = cell(numPipelines, length(varNamesForDetailedTable));
for iPipeline = 1:numPipelines
    dataForDetailedTable{iPipeline, 1} = pipelineNamesList{iPipeline};
    idxOffset = 1;
    % OR Metrics
    if isfield(overallComparisonResults, 'Strategy_OR') && ~isempty(overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}) && isfield(overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}, 'outerFoldMetrics_mean')
        for mIdx = 1:length(metricNamesFull)
            dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 1} = overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}.outerFoldMetrics_mean(mIdx);
            dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 2} = overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}.outerFoldMetrics_std(mIdx);
        end
    else
         for mIdx = 1:length(metricNamesFull), dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 1} = NaN; dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 2} = NaN; end
    end
    idxOffset = 1 + 2*length(metricNamesFull);
    % AND Metrics
    if isfield(overallComparisonResults, 'Strategy_AND') && ~isempty(overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}) && isfield(overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}, 'outerFoldMetrics_mean')
        for mIdx = 1:length(metricNamesFull)
            dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 1} = overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}.outerFoldMetrics_mean(mIdx);
            dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 2} = overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}.outerFoldMetrics_std(mIdx);
        end
    else
         for mIdx = 1:length(metricNamesFull), dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 1} = NaN; dataForDetailedTable{iPipeline, idxOffset + (mIdx-1)*2 + 2} = NaN; end
    end
end
detailedComparisonCSVTable = cell2table(dataForDetailedTable, 'VariableNames', varNamesForDetailedTable);
disp('Comprehensive Comparison Table:');
disp(detailedComparisonCSVTable);
comparisonTableFilename = fullfile(resultsPath, '..', 'OutlierStrategyComparison_Results', ... % Save one level up, new subfolder
                                   sprintf('%s_OutlierStrategies_AllMetricsComparison.csv', dateStrForFilenames));
if ~exist(fileparts(comparisonTableFilename), 'dir'), mkdir(fileparts(comparisonTableFilename)); end
writetable(detailedComparisonCSVTable, comparisonTableFilename);
fprintf('Comprehensive comparison CSV table saved to: %s\n', comparisonTableFilename);


% --- Spider Plot Comparison (Optional, if spider_plot.m is available) ---
if exist('spider_plot', 'file') && exist('spider_plot_R2019b', 'file') % Check for both versions for robustness
    fprintf('\nAttempting to generate Spider Plot for comparison...\n');
    metricsForSpider = {'F2_WHO3', 'Sensitivity_WHO3', 'Specificity_WHO1', 'AUC', 'Accuracy'}; % Choose key metrics
    
    spiderData_P_list = {}; % Cell array to hold data for each pipeline
    spiderLegendLabels = {};
    
    axesLimitsSpider = []; % Determine dynamically or set fixed e.g. [0;1]
    
    pipelineNamesForSpider = {};

    for iPipeline = 1:numPipelines
        tempDataOR  = [];
        tempDataAND = [];
        validPipeline = true;

        for iMetric = 1:length(metricsForSpider)
            currentSpiderMetric = metricsForSpider{iMetric};
            metricSpiderIdx = find(strcmpi(metricNamesFull, currentSpiderMetric));
            if isempty(metricSpiderIdx)
                warning('Metric %s not found for spider plot for pipeline %s. Skipping metric.', currentSpiderMetric, pipelineNamesList{iPipeline});
                tempDataOR = [tempDataOR, NaN]; % Add NaN to keep structure
                tempDataAND = [tempDataAND, NaN];
                continue;
            end
            
            valOR = NaN; valAND = NaN;
            if isfield(overallComparisonResults, 'Strategy_OR') && ~isempty(overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}) && isfield(overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}, 'outerFoldMetrics_mean')
                valOR = overallComparisonResults.Strategy_OR.allPipelinesResults{iPipeline}.outerFoldMetrics_mean(metricSpiderIdx);
            end
            if isfield(overallComparisonResults, 'Strategy_AND') && ~isempty(overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}) && isfield(overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}, 'outerFoldMetrics_mean')
                valAND = overallComparisonResults.Strategy_AND.allPipelinesResults{iPipeline}.outerFoldMetrics_mean(metricSpiderIdx);
            end
            
            if isnan(valOR) || isnan(valAND)
                % If any metric is NaN for a pipeline under either strategy, maybe skip this pipeline for spider plot
                % validPipeline = false; break; 
            end
            tempDataOR = [tempDataOR, valOR];
            tempDataAND = [tempDataAND, valAND];
        end
        
        if validPipeline && ~all(isnan(tempDataOR)) && ~all(isnan(tempDataAND)) % Add if not all NaN
            spiderData_P_list{end+1} = [tempDataOR; tempDataAND]; % Each pipeline will have 2 rows (OR, AND)
            pipelineNamesForSpider{end+1} = pipelineNamesList{iPipeline}; % Store name of valid pipeline
        end
    end
    
    if ~isempty(spiderData_P_list)
        % Now, create a single P matrix for spider_plot where rows are strategy-pipeline combinations
        % or plot multiple spider_plots (one per pipeline, comparing OR vs AND)
        % Let's do one per pipeline for clarity
        
        spiderAxesLabels = strrep(metricsForSpider, '_', ' ');
        
        % Determine common axes limits for spider plots, e.g. 0 to 1
        commonSpiderAxesLimits = repmat([0; 1], 1, length(metricsForSpider));

        for iPlotPipeline = 1:length(spiderData_P_list)
            P_spider_pipeline = spiderData_P_list{iPlotPipeline};
            P_spider_pipeline(isnan(P_spider_pipeline)) = 0; % Replace NaN with 0 for plotting if necessary

            figSpiderPipe = figure('Name', sprintf('Spider Plot - %s', pipelineNamesForSpider{iPlotPipeline}));
            try
                spider_plot_R2019b(P_spider_pipeline, ... % Use R2019b version if available
                            'AxesLabels', spiderAxesLabels, ...
                            'AxesLimits', commonSpiderAxesLimits, ...
                            'FillOption', 'on', ...
                            'FillTransparency', [0.2, 0.1], ...
                            'Color', [[0.9,0.6,0.4]; [0.4,0.702,0.902]], ... % Orange for OR, Blue for AND
                            'LineWidth', 1.5, ...
                            'Marker', {'o', 's'}, ...
                            'MarkerSize', 60);
                title(sprintf('Performance Comparison for %s', pipelineNamesForSpider{iPlotPipeline}));
                legend({'T2 OR Q Strategy', 'T2 AND Q Strategy'}, 'Location', 'bestoutside');
                
                spiderPlotPipeFilenameBase = fullfile(comparisonFiguresPath, sprintf('%s_SpiderPlot_%s_StratCompare', dateStrForFilenames, pipelineNamesForSpider{iPlotPipeline}));
                savefig(figSpiderPipe, [spiderPlotPipeFilenameBase, '.fig']);
                exportgraphics(figSpiderPipe, [spiderPlotPipeFilenameBase, '.tiff'], 'Resolution', 300);
                fprintf('Spider plot for pipeline %s saved.\n', pipelineNamesForSpider{iPlotPipeline});
            catch ME_spider_pipe
                fprintf('Error generating spider plot for pipeline %s: %s\n', pipelineNamesForSpider{iPlotPipeline}, ME_spider_pipe.message);
            end
            if iPlotPipeline >= 6 % Limit number of individual spider plots for brevity
                fprintf('Stopping individual spider plots after 6 for brevity.\n');
                break;
            end
        end
    else
        fprintf('No valid data for spider plots after filtering or metric selection.\n');
    end
else
    fprintf('spider_plot.m or spider_plot_R2019b.m not found. Skipping spider plot visualization.\n');
end


fprintf('\nPHASE 2 Processing Complete (Comprehensive Outlier Strategy Comparison): %s\n', string(datetime('now')));