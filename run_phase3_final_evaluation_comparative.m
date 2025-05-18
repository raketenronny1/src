% run_phase3_final_evaluation_comparative.m % << NEW FILENAME (Suggestion)
%
% Script for Phase 3: Final Model Training & Unbiased Evaluation.
% MODIFIED to run in parallel for "T2 OR Q" and "T2 AND Q" outlier strategies.
% 1. For each strategy:
%    a. Loads the corresponding cleaned training set.
%    b. Determines (or uses predefined) best hyperparameters for MRMRLDA.
%    c. Trains the MRMRLDA pipeline on the entire strategy-specific training set.
%    d. Evaluates the final model on the common unseen test set.
% 2. Saves results and models separately for each strategy.
%
% Date: 2025-05-18 (Modified for parallel strategy evaluation)

%% 0. Initialization
% =========================================================================
clear; clc; close all;
fprintf('PHASE 3: Final Model Training & Unbiased Evaluation (Comparative Strategies) - %s\n', string(datetime('now')));

% --- Define Paths (Simplified) ---
projectRoot = pwd; 
if ~exist(fullfile(projectRoot, 'src'), 'dir') || ~exist(fullfile(projectRoot, 'data'), 'dir')
    error(['Project structure not found. Please ensure MATLAB''s "Current Folder" is set to your ' ...
           'main project root directory before running. Current directory is: %s'], projectRoot);
end

srcPath       = fullfile(projectRoot, 'src');
helperFunPath = fullfile(srcPath, 'helper_functions');
if ~exist(helperFunPath, 'dir')
    error('The ''helper_functions'' directory was not found inside ''%s''.', srcPath);
end
addpath(helperFunPath);

dataPath      = fullfile(projectRoot, 'data');
phase2ResultsPath = fullfile(projectRoot, 'results', 'Phase2'); % For loading best hyperparameters
resultsPath_P3   = fullfile(projectRoot, 'results', 'Phase3_Comparative'); % Specific to this comparative Phase 3
modelsPath_P3    = fullfile(projectRoot, 'models', 'Phase3_Comparative');   % Specific to this comparative Phase 3
figuresPath_P3   = fullfile(projectRoot, 'figures', 'Phase3_Comparative'); % Specific to this comparative Phase 3

if ~exist(resultsPath_P3, 'dir'), mkdir(resultsPath_P3); end
if ~exist(modelsPath_P3, 'dir'), mkdir(modelsPath_P3); end
if ~exist(figuresPath_P3, 'dir'), mkdir(figuresPath_P3); end

dateStr = string(datetime('now','Format','yyyyMMdd'));

% Define Outlier Strategies to process
outlierStrategies = {'OR', 'AND'};
overallPhase3Results = struct(); % To store key results for final comparison

%% --- MAIN LOOP FOR OUTLIER STRATEGIES ---
for iStrat = 1:length(outlierStrategies)
    currentStrategy = outlierStrategies{iStrat};
    fprintf('\n\n====================================================================\n');
    fprintf('   PROCESSING PHASE 3 FOR OUTLIER STRATEGY: T2 %s Q\n', currentStrategy);
    fprintf('====================================================================\n');

    %% 1. Load Data & Best Pipeline Info for Current Strategy
    % =========================================================================
    
    % --- Load Wavenumbers (common for all) ---
    try
        wavenumbers_data = load(fullfile(dataPath, 'wavenumbers.mat'), 'wavenumbers_roi');
        wavenumbers_original = wavenumbers_data.wavenumbers_roi;
        if iscolumn(wavenumbers_original), wavenumbers_original = wavenumbers_original'; end
    catch ME_wave
        fprintf('ERROR loading wavenumbers.mat: %s\n', ME_wave.message);
        continue; % Skip to next strategy if wavenumbers can't be loaded
    end

    % --- Load Strategy-Specific Entire Training Set ---
    fprintf('Loading strategy-specific training set (%s)...\n', currentStrategy);
    if strcmp(currentStrategy, 'OR')
        trainingDataFilePattern = '*_training_set_no_outliers_T2orQ.mat';
        varName_X = 'X_train_no_outliers_OR';
        varName_y = 'y_train_no_outliers_OR_num'; % Assuming numeric labels
    elseif strcmp(currentStrategy, 'AND')
        trainingDataFilePattern = '*_training_set_no_outliers_T2andQ.mat';
        varName_X = 'X_train_no_outliers_AND';
        varName_y = 'y_train_no_outliers_AND_num'; % Assuming numeric labels
    else
        error('Unknown strategy in loop.');
    end
    
    trainingDataFiles = dir(fullfile(dataPath, trainingDataFilePattern));
    if isempty(trainingDataFiles)
        warning('No training set file found for strategy "%s" in %s. Skipping this strategy.', currentStrategy, dataPath);
        continue;
    end
    [~,idxSortTrain] = sort([trainingDataFiles.datenum],'descend');
    latestTrainingDataFile = fullfile(dataPath, trainingDataFiles(idxSortTrain(1)).name);
    
    try
        loadedTrainingData = load(latestTrainingDataFile, varName_X, varName_y); % Add other vars if needed like Patient_ID
        X_train_full = loadedTrainingData.(varName_X);
        y_train_full = loadedTrainingData.(varName_y);
        % Ensure y_train_full is numeric and column vector
        if ~isnumeric(y_train_full), error('y_train_full for strategy %s is not numeric.', currentStrategy); end
        y_train_full = y_train_full(:);

        fprintf('Training data for strategy %s loaded: %d spectra, %d features from %s.\n', ...
            currentStrategy, size(X_train_full, 1), size(X_train_full, 2), latestTrainingDataFile);
        if isempty(X_train_full), error('X_train_full is empty for strategy %s.', currentStrategy); end
    catch ME_load_train
        fprintf('ERROR loading training data for strategy %s from %s: %s\n', currentStrategy, latestTrainingDataFile, ME_load_train.message);
        continue; 
    end

    % --- Load Test Set (common for all strategies) ---
    % This only needs to be done once if it's outside the loop, or ensure it's correctly handled if inside.
    % For simplicity, loading it inside, though it's redundant if test set is always the same.
    fprintf('Loading common test set...\n');
    testDataFile = fullfile(dataPath, 'data_table_test.mat'); % This should be the original, unprocessed test set table
    try
        loadedTestData = load(testDataFile, 'dataTableTest');
        dataTableTest = loadedTestData.dataTableTest;

        numTestProbes = height(dataTableTest);
        temp_X_test_list = cell(numTestProbes, 1);
        temp_y_test_list = cell(numTestProbes, 1);
        temp_probeIDs_test_list = cell(numTestProbes, 1);
        
        totalTestSpectra = 0;
        for i = 1:numTestProbes
            % Assuming CombinedSpectra in dataTableTest contains the raw spectra that need the same preprocessing
            % as applied to the training data (e.g. SG, SNV, L2-norm if done prior to PCA for outlier removal)
            % OR if CombinedSpectra already contains fully preprocessed spectra.
            % For now, assume CombinedSpectra is ready for binning/feature selection as per the model.
            spectraMatrix = dataTableTest.CombinedSpectra{i,1}; 
            numSpectraThisProbe = size(spectraMatrix, 1);
            totalTestSpectra = totalTestSpectra + numSpectraThisProbe;
            temp_X_test_list{i} = spectraMatrix;
            current_WHO_grade_cat = dataTableTest.WHO_Grade(i);
            if current_WHO_grade_cat == 'WHO-1', temp_y_test_list{i} = ones(numSpectraThisProbe, 1) * 1;
            elseif current_WHO_grade_cat == 'WHO-3', temp_y_test_list{i} = ones(numSpectraThisProbe, 1) * 3;
            else, temp_y_test_list{i} = ones(numSpectraThisProbe, 1) * NaN; end
            temp_probeIDs_test_list{i} = repmat(dataTableTest.Diss_ID(i), numSpectraThisProbe, 1);
        end
        
        X_test_full = vertcat(temp_X_test_list{:});
        y_test_full_numeric = vertcat(temp_y_test_list{:});
        y_test_full_numeric = y_test_full_numeric(:); % Ensure column
        probeIDs_test_full = vertcat(temp_probeIDs_test_list{:});

        nan_label_idx_test = isnan(y_test_full_numeric);
        if any(nan_label_idx_test)
            fprintf('Removing %d test spectra with NaN labels.\n', sum(nan_label_idx_test));
            X_test_full(nan_label_idx_test,:) = [];
            y_test_full_numeric(nan_label_idx_test) = [];
            probeIDs_test_full(nan_label_idx_test) = [];
        end
        fprintf('Test data loaded and processed: %d spectra, %d features from %d probes.\n', ...
                size(X_test_full, 1), size(X_test_full, 2), numTestProbes);
        if size(X_test_full,2) ~= size(X_train_full,2) && ~isempty(X_train_full) % Check if X_train_full was loaded
            error('Mismatch in number of features between training (%d) and test (%d) data before binning.', size(X_train_full,2), size(X_test_full,2));
        end
    catch ME_load_test
        fprintf('ERROR loading or processing test data from %s: %s\n', testDataFile, ME_load_test.message);
        continue;
    end

    % --- Define/Load Final Model Hyperparameters for MRMRLDA for the CURRENT STRATEGY ---
    % OPTION A: Hardcode (as in original script, but less flexible)
    % final_binningFactor = 8;
    % final_numMRMRFeatures = 50;
    % fprintf('Using PREDEFINED hyperparameters for MRMRLDA: Binning Factor = %d, Num MRMR Features = %d\n', ...
    %    final_binningFactor, final_numMRMRFeatures);

    % OPTION B: Load best hyperparameters from Phase 2 for the current strategy
    fprintf('Loading best hyperparameters for MRMRLDA (Strategy: %s) from Phase 2 results...\n', currentStrategy);
    bestHyperparamFile = fullfile(phase2ResultsPath, sprintf('*_Phase2_BestPipelineInfo_Strat_%s.mat', currentStrategy));
    bestHyperparamFiles = dir(bestHyperparamFile);
    if isempty(bestHyperparamFiles)
        warning('No Phase 2 best hyperparameter file found for MRMRLDA and strategy %s. Using predefined defaults.', currentStrategy);
        final_binningFactor = 8; % Default
        final_numMRMRFeatures = 50; % Default
    else
        [~,idxSortHP] = sort([bestHyperparamFiles.datenum],'descend');
        latestBestHPFile = fullfile(phase2ResultsPath, bestHyperparamFiles(idxSortHP(1)).name);
        try
            loadedHP = load(latestBestHPFile, 'bestPipelineSummary_strat');
            % Assuming 'MRMRLDA' was the best. If not, this logic needs to find it.
            % And assuming 'bestPipelineSummary_strat.outerFoldBestHyperparams' contains the chosen ones.
            % For simplicity, let's assume 'bestPipelineSummary_strat.pipelineConfig' holds the final single choice from Phase 2,
            % or we extract the mode of outerFoldBestHyperparams.
            % The current `run_phase2_model_selection_comparative.m` saves `bestPipelineSummary_strat`
            % which has `outerFoldBestHyperparams` (a cell array, one per outer fold).
            % We need to determine the *single* final hyperparameter set.
            % Let's take the mode of the selected binning factors and MRMR features from the outer folds.

            all_binning_factors_p2 = [];
            all_mrmr_features_p2 = [];
            for k_hp = 1:length(loadedHP.bestPipelineSummary_strat.outerFoldBestHyperparams)
                if isfield(loadedHP.bestPipelineSummary_strat.outerFoldBestHyperparams{k_hp}, 'binningFactor')
                    all_binning_factors_p2 = [all_binning_factors_p2; loadedHP.bestPipelineSummary_strat.outerFoldBestHyperparams{k_hp}.binningFactor];
                end
                if isfield(loadedHP.bestPipelineSummary_strat.outerFoldBestHyperparams{k_hp}, 'numMRMRFeatures')
                    all_mrmr_features_p2 = [all_mrmr_features_p2; loadedHP.bestPipelineSummary_strat.outerFoldBestHyperparams{k_hp}.numMRMRFeatures];
                end
            end
            
            if ~isempty(all_binning_factors_p2)
                final_binningFactor = mode(all_binning_factors_p2);
            else
                warning('No binning factors found in Phase 2 best hyperparams for %s. Using default 8.', currentStrategy);
                final_binningFactor = 8;
            end
            if ~isempty(all_mrmr_features_p2)
                final_numMRMRFeatures = mode(all_mrmr_features_p2);
            else
                warning('No numMRMRFeatures found in Phase 2 best hyperparams for %s. Using default 50.', currentStrategy);
                final_numMRMRFeatures = 50;
            end
             fprintf('Using hyperparameters for MRMRLDA (Strategy %s) from Phase 2: Binning=%d, MRMR Feats=%d (Mode from outer folds of best pipeline)\n', ...
                 currentStrategy, final_binningFactor, final_numMRMRFeatures);

        catch ME_loadHP
            fprintf('ERROR loading best hyperparameters for strategy %s: %s. Using predefined defaults.\n', currentStrategy, ME_loadHP.message);
            final_binningFactor = 8; % Default
            final_numMRMRFeatures = 50; % Default
        end
    end
    
    % --- Define Metric Names ---
    metricNames = {'Accuracy', 'Sensitivity_WHO3', 'Specificity_WHO1', 'PPV_WHO3', 'NPV_WHO1', 'F1_WHO3', 'F2_WHO3', 'AUC'};

    %% 2. Train Final Model on Entire Strategy-Specific Training Set (MRMRLDA)
    % =========================================================================
    fprintf('\n--- Training Final MRMRLDA Model on Entire Training Set (Strategy: %s) ---\n', currentStrategy);

    % Apply Binning
    if final_binningFactor > 1
        [X_train_binned, wavenumbers_binned] = bin_spectra(X_train_full, wavenumbers_original, final_binningFactor);
    else
        X_train_binned = X_train_full;
        wavenumbers_binned = wavenumbers_original;
    end
    
    % Apply MRMR
    y_train_cat = categorical(y_train_full); 
    final_selected_feature_indices_in_binned_space = [];
    final_selected_wavenumbers = [];
    X_train_fs = X_train_binned; % Default if no MRMR features or selection fails

    if final_numMRMRFeatures > 0 && size(X_train_binned, 2) > 0 && final_numMRMRFeatures <= size(X_train_binned,2)
        try
            [ranked_indices_all_train, ~] = fscmrmr(X_train_binned, y_train_cat);
            actual_num_to_select_final = min(final_numMRMRFeatures, length(ranked_indices_all_train));
            if actual_num_to_select_final < final_numMRMRFeatures
                warning('MRMR: Requested %d features, but only %d available/selected from %s training set. Using %d.', ...
                        final_numMRMRFeatures, length(ranked_indices_all_train), currentStrategy, actual_num_to_select_final);
            end
            if actual_num_to_select_final > 0
                final_selected_feature_indices_in_binned_space = ranked_indices_all_train(1:actual_num_to_select_final);
                final_selected_wavenumbers = wavenumbers_binned(final_selected_feature_indices_in_binned_space);
                X_train_fs = X_train_binned(:, final_selected_feature_indices_in_binned_space);
            else
                 warning('MRMR resulted in 0 features for strategy %s. Using all binned features for LDA.', currentStrategy);
            end
        catch ME_mrmr_final
            fprintf('ERROR during final MRMR for strategy %s: %s. Using all binned features.\n', currentStrategy, ME_mrmr_final.message);
        end
    elseif final_numMRMRFeatures > size(X_train_binned,2)
        warning('Requested numMRMRFeatures (%d) > available binned features (%d) for strategy %s. Using all binned features.', ...
            final_numMRMRFeatures, size(X_train_binned,2), currentStrategy);
    end
    
    fprintf('MRMR selected %d features. Final training data for LDA (Strategy %s): %d spectra, %d features.\n', ...
        length(final_selected_feature_indices_in_binned_space), currentStrategy, size(X_train_fs,1), size(X_train_fs,2));

    % Train LDA
    if isempty(X_train_fs) || size(X_train_fs,1) < 2 || length(unique(y_train_full)) < 2
        error('Insufficient data for final LDA training for strategy %s.', currentStrategy);
    end
    final_LDAModel_strat = fitcdiscr(X_train_fs, y_train_full);
    fprintf('Final LDA model trained for strategy %s.\n', currentStrategy);

    %% 3. Evaluate Final Model on Test Set for Current Strategy
    % =========================================================================
    fprintf('\n--- Evaluating Final Model on Unseen Test Set (Strategy: %s) ---\n', currentStrategy);
    
    % Apply Binning to Test Set
    if final_binningFactor > 1
        [X_test_binned, ~] = bin_spectra(X_test_full, wavenumbers_original, final_binningFactor);
    else
        X_test_binned = X_test_full;
    end
    
    % Apply Feature Selection to Test Set (using indices from training)
    X_test_fs = X_test_binned; % Default if no features were selected
    if ~isempty(final_selected_feature_indices_in_binned_space)
        X_test_fs = X_test_binned(:, final_selected_feature_indices_in_binned_space);
    end
    
    % Predict on Test Set
    [y_pred_test_strat, y_scores_test_strat] = predict(final_LDAModel_strat, X_test_fs);
    
    % Calculate Spectrum-Level Performance Metrics
    positiveClassLabel = 3; 
    classOrder_strat = final_LDAModel_strat.ClassNames;
    positiveClassColIdx_strat = find(classOrder_strat == positiveClassLabel); % Simplified
    if isempty(positiveClassColIdx_strat) || max(positiveClassColIdx_strat) > size(y_scores_test_strat,2)
        error('Positive class for AUC not found correctly for strategy %s test set.', currentStrategy);
    end
    scores_for_positive_class_test_strat = y_scores_test_strat(:, positiveClassColIdx_strat);
    testSetPerformanceMetrics_spectrum_strat = calculate_performance_metrics(y_test_full_numeric, y_pred_test_strat, scores_for_positive_class_test_strat, positiveClassLabel, metricNames);
    fprintf('\n--- Test Set Performance Metrics (Spectrum-Level, Strategy: %s) ---\n', currentStrategy);
    disp(struct2table(testSetPerformanceMetrics_spectrum_strat));
    
    %% 4. Probe-Level Aggregation and Evaluation for Current Strategy
    % =========================================================================
    fprintf('\n--- Aggregating results to Probe-Level for Test Set (Strategy: %s) ---\n', currentStrategy);
    uniqueTestProbes_strat = unique(probeIDs_test_full, 'stable');
    numUniqueTestProbes_strat = length(uniqueTestProbes_strat);
    probeLevelResults_strat = table('Size', [numUniqueTestProbes_strat, 7], ...
        'VariableTypes', {'cell', 'double', 'categorical', 'double', 'double', 'double', 'double'}, ...
        'VariableNames', {'Diss_ID', 'True_WHO_Grade_Numeric', 'True_WHO_Grade_Category', ...
                          'Mean_WHO3_Probability', 'Predicted_WHO_Grade_Numeric_MeanProb', ...
                          'MajorityVote_Predicted_WHO_Grade_Numeric', 'Proportion_Spectra_Predicted_WHO3'});
    probeLevelResults_strat.Diss_ID = uniqueTestProbes_strat;

    for i_probe = 1:numUniqueTestProbes_strat
        currentProbeID_strat = uniqueTestProbes_strat{i_probe};
        idxSpectraForProbe_strat = strcmp(probeIDs_test_full, currentProbeID_strat);
        
        true_labels_this_probe_strat = y_test_full_numeric(idxSpectraForProbe_strat);
        if ~isempty(true_labels_this_probe_strat)
            probeLevelResults_strat.True_WHO_Grade_Numeric(i_probe) = mode(true_labels_this_probe_strat);
            if mode(true_labels_this_probe_strat) == 1, probeLevelResults_strat.True_WHO_Grade_Category(i_probe) = 'WHO-1';
            elseif mode(true_labels_this_probe_strat) == 3, probeLevelResults_strat.True_WHO_Grade_Category(i_probe) = 'WHO-3'; end
        end
        
        mean_prob_who3_strat = mean(scores_for_positive_class_test_strat(idxSpectraForProbe_strat));
        probeLevelResults_strat.Mean_WHO3_Probability(i_probe) = mean_prob_who3_strat;
        otherClasses_strat = setdiff(unique(y_train_full), positiveClassLabel); 
        defaultOtherClass_strat = 1; if ~isempty(otherClasses_strat), defaultOtherClass_strat = otherClasses_strat(1); end
        if mean_prob_who3_strat > 0.5, probeLevelResults_strat.Predicted_WHO_Grade_Numeric_MeanProb(i_probe) = positiveClassLabel;
        else, probeLevelResults_strat.Predicted_WHO_Grade_Numeric_MeanProb(i_probe) = defaultOtherClass_strat; end
        
        predicted_labels_this_probe_strat = y_pred_test_strat(idxSpectraForProbe_strat);
        probeLevelResults_strat.MajorityVote_Predicted_WHO_Grade_Numeric(i_probe) = mode(predicted_labels_this_probe_strat);
        probeLevelResults_strat.Proportion_Spectra_Predicted_WHO3(i_probe) = sum(predicted_labels_this_probe_strat == positiveClassLabel) / length(predicted_labels_this_probe_strat);
    end
    fprintf('\nProbe-Level Aggregated Results (Test Set, Strategy: %s):\n', currentStrategy);
    disp(probeLevelResults_strat);

    % Calculate Probe-Level Performance Metrics
    y_true_probe_strat = probeLevelResults_strat.True_WHO_Grade_Numeric;
    y_pred_probe_mean_prob_strat = probeLevelResults_strat.Predicted_WHO_Grade_Numeric_MeanProb;
    probe_scores_for_positive_class_strat = probeLevelResults_strat.Mean_WHO3_Probability;
    valid_probes_idx_strat = ~isnan(y_true_probe_strat) & ~isnan(y_pred_probe_mean_prob_strat);

    if sum(valid_probes_idx_strat) > 0 && length(unique(y_true_probe_strat(valid_probes_idx_strat))) > 1
        fprintf('\n--- Probe-Level Performance Metrics (based on Mean Probability > 0.5, Strategy: %s) ---\n', currentStrategy);
        probeLevelPerfMetrics_strat = calculate_performance_metrics(y_true_probe_strat(valid_probes_idx_strat), ...
            y_pred_probe_mean_prob_strat(valid_probes_idx_strat), probe_scores_for_positive_class_strat(valid_probes_idx_strat), positiveClassLabel, metricNames);
        disp(struct2table(probeLevelPerfMetrics_strat));
    else
        fprintf('\nCould not calculate probe-level performance metrics for %s (e.g., only one class present or no valid probes).\n', currentStrategy);
        probeLevelPerfMetrics_strat = struct(); 
        for m=1:length(metricNames), probeLevelPerfMetrics_strat.(metricNames{m}) = NaN; end % Fill with NaNs
    end
    
    % Store results for this strategy
    overallPhase3Results.(sprintf('Strategy_%s', currentStrategy)).testSetPerformance_Spectrum = testSetPerformanceMetrics_spectrum_strat;
    overallPhase3Results.(sprintf('Strategy_%s', currentStrategy)).probeLevelResults = probeLevelResults_strat;
    overallPhase3Results.(sprintf('Strategy_%s', currentStrategy)).probeLevelPerformance_MeanProb = probeLevelPerfMetrics_strat;

    %% 5. Save Final Model and Results for Current Strategy
    % =========================================================================
    finalModelPackage_strat = struct();
    finalModelPackage_strat.description = sprintf('Final MRMRLDA model (Strategy: %s) trained on the full corresponding training set.', currentStrategy);
    finalModelPackage_strat.trainingDate = string(datetime('now'));
    finalModelPackage_strat.LDAModel = final_LDAModel_strat;
    finalModelPackage_strat.binningFactor = final_binningFactor;
    finalModelPackage_strat.numMRMRFeaturesSelected = length(final_selected_feature_indices_in_binned_space);
    finalModelPackage_strat.selectedFeatureIndices_in_binned_space = final_selected_feature_indices_in_binned_space;
    finalModelPackage_strat.selectedWavenumbers = final_selected_wavenumbers;
    finalModelPackage_strat.originalWavenumbers_before_binning = wavenumbers_original;
    finalModelPackage_strat.binnedWavenumbers_for_selection = wavenumbers_binned; % Wavenumbers after binning, before MRMR
    finalModelPackage_strat.testSetPerformance_Spectrum = testSetPerformanceMetrics_spectrum_strat;
    finalModelPackage_strat.probeLevelResults = probeLevelResults_strat;
    finalModelPackage_strat.probeLevelPerformance_MeanProb = probeLevelPerfMetrics_strat;
    finalModelPackage_strat.trainingDataFile = latestTrainingDataFile; 
    finalModelPackage_strat.testDataFile = testDataFile;
    finalModelPackage_strat.outlierStrategyUsed = currentStrategy;

    modelFilename_strat = fullfile(modelsPath_P3, sprintf('%s_Phase3_FinalMRMRLDA_Model_Strat_%s.mat', dateStr, currentStrategy));
    save(modelFilename_strat, 'finalModelPackage_strat'); % Save as 'finalModelPackage_strat'
    fprintf('\nFinal model package for strategy %s saved to: %s\n', currentStrategy, modelFilename_strat);

    resultsFilename_phase3_strat = fullfile(resultsPath_P3, sprintf('%s_Phase3_TestSetResults_Strat_%s.mat', dateStr, currentStrategy));
    save(resultsFilename_phase3_strat, 'testSetPerformanceMetrics_spectrum_strat', 'probeLevelResults_strat', 'probeLevelPerfMetrics_strat', ...
         'final_binningFactor', 'final_numMRMRFeatures', 'final_selected_wavenumbers', 'currentStrategy');
    fprintf('Phase 3 test set results for strategy %s saved to: %s\n', currentStrategy, resultsFilename_phase3_strat);

    %% 6. Visualizations for Current Strategy (Confusion Matrices, Probe Plots)
    % =========================================================================
    % Spectrum-Level Confusion Matrix
    if exist('confusionchart','file') && ~isempty(y_test_full_numeric) && ~isempty(y_pred_test_strat)
        figConfSpec = figure('Name', sprintf('Confusion Matrix (Spectrum, Test Set) - %s Strategy', currentStrategy));
        cm_spec = confusionchart(y_test_full_numeric, y_pred_test_strat, ...
            'ColumnSummary','column-normalized', 'RowSummary','row-normalized', ...
            'Title', sprintf('CM Spectrum (Test) - Strat: %s (F2: %.3f)', currentStrategy, testSetPerformanceMetrics_spectrum_strat.F2_WHO3));
        confMatFigFilenameBase_spec = fullfile(figuresPath_P3, sprintf('%s_Phase3_ConfMat_Spectrum_Strat_%s', dateStr, currentStrategy));
        savefig(figConfSpec, [confMatFigFilenameBase_spec, '.fig']);
        exportgraphics(figConfSpec, [confMatFigFilenameBase_spec, '.tiff'], 'Resolution', 300);
        close(figConfSpec);
    end
    
    % Probe-Level Confusion Matrix (Mean Probability)
    if exist('confusionchart','file') && sum(valid_probes_idx_strat) > 0 && length(unique(y_true_probe_strat(valid_probes_idx_strat))) > 1
        figConfProbe = figure('Name', sprintf('Confusion Matrix (Probe, Mean Prob, Test Set) - %s Strategy', currentStrategy));
        cm_probe = confusionchart(y_true_probe_strat(valid_probes_idx_strat), y_pred_probe_mean_prob_strat(valid_probes_idx_strat), ...
            'ColumnSummary','column-normalized', 'RowSummary','row-normalized', ...
            'Title', sprintf('CM Probe (MeanProb, Test) - Strat: %s (F2: %.3f)', currentStrategy, probeLevelPerfMetrics_strat.F2_WHO3));
        confMatProbeFigFilenameBase = fullfile(figuresPath_P3, sprintf('%s_Phase3_ConfMat_Probe_MeanProb_Strat_%s', dateStr, currentStrategy));
        savefig(figConfProbe, [confMatProbeFigFilenameBase, '.fig']);
        exportgraphics(figConfProbe, [confMatProbeFigFilenameBase, '.tiff'], 'Resolution', 300);
        close(figConfProbe);
    end
    
    % Probe-Level Probability Plot
    figProbDist_strat = figure('Name', sprintf('Probe-Level Mean WHO-3 Probabilities (Test Set) - %s Strategy', currentStrategy), 'Position', [100, 100, 900, 700]);
    hold on; jitterAmount = 0.02; 
    probes_true_who1_plot = probeLevelResults_strat(probeLevelResults_strat.True_WHO_Grade_Numeric == 1, :);
    probes_true_who3_plot = probeLevelResults_strat(probeLevelResults_strat.True_WHO_Grade_Numeric == 3, :);
    h_p3_1_strat = []; h_p3_3_strat = [];
    if ~isempty(probes_true_who1_plot), x_coords_who1 = 1 + (rand(height(probes_true_who1_plot),1)-0.5)*jitterAmount*2; h_p3_1_strat = scatter(x_coords_who1, probes_true_who1_plot.Mean_WHO3_Probability,70,'o','MarkerEdgeColor','k','MarkerFaceColor',colorWHO1,'LineWidth',1,'DisplayName','True WHO-1 Probes'); end
    if ~isempty(probes_true_who3_plot), x_coords_who3 = 2 + (rand(height(probes_true_who3_plot),1)-0.5)*jitterAmount*2; h_p3_3_strat = scatter(x_coords_who3, probes_true_who3_plot.Mean_WHO3_Probability,70,'s','MarkerEdgeColor','k','MarkerFaceColor',colorWHO3,'LineWidth',1,'DisplayName','True WHO-3 Probes'); end
    plot([0.5 2.5], [0.5 0.5], 'k--', 'DisplayName', 'Decision Threshold (0.5)');
    hold off; xticks([1 2]); xticklabels({'True WHO-1', 'True WHO-3'}); xlim([0.5 2.5]); ylim([0 1]);
    ylabel('Mean Predicted Probability of WHO-3'); title(sprintf('Probe-Level Classification Probabilities (Test Set) - %s Strategy', currentStrategy)); grid on;
    if ~isempty(h_p3_1_strat) || ~isempty(h_p3_3_strat), legend([h_p3_1_strat, h_p3_3_strat], 'Location', 'best'); end; set(gca, 'FontSize', 12);
    probeProbFigFilenameBase_strat = fullfile(figuresPath_P3, sprintf('%s_Phase3_ProbeLevelProbabilities_Strat_%s', dateStr, currentStrategy));
    savefig(figProbDist_strat, [probeProbFigFilenameBase_strat, '.fig']);
    exportgraphics(figProbDist_strat, [probeProbFigFilenameBase_strat, '.tiff'], 'Resolution', 300);
    close(figProbDist_strat);

    % Probe-Level Violin Plot (can be adapted similarly)
    % ... (code for violin plot from original script, add strategy to title and filename) ...
    % For brevity, this is omitted here but would follow the pattern above.

end % End of MAIN LOOP FOR OUTLIER STRATEGIES

%% 7. Final Comparison of Test Set Performances
% =========================================================================
fprintf('\n\n====================================================================\n');
fprintf('   FINAL TEST SET PERFORMANCE COMPARISON (MRMRLDA Pipeline)\n');
fprintf('====================================================================\n');

comp_table_data = cell(length(metricNames)+1, length(outlierStrategies)+1);
comp_table_data(1,1) = {'Metric'};
for i=1:length(outlierStrategies), comp_table_data(1,i+1) = {sprintf('TestSet Perf (%s)', outlierStrategies{i})}; end

for m=1:length(metricNames)
    comp_table_data(m+1,1) = metricNames(m);
    for s=1:length(outlierStrategies)
        strat_name_field = sprintf('Strategy_%s', outlierStrategies{s});
        % Compare spectrum-level performance for this example
        if isfield(overallPhase3Results, strat_name_field) && ...
           isfield(overallPhase3Results.(strat_name_field), 'testSetPerformance_Spectrum') && ...
           isfield(overallPhase3Results.(strat_name_field).testSetPerformance_Spectrum, metricNames{m})
            
            val = overallPhase3Results.(strat_name_field).testSetPerformance_Spectrum.(metricNames{m});
            comp_table_data(m+1,s+1) = {sprintf('%.4f', val)};
        else
            comp_table_data(m+1,s+1) = {'N/A'};
        end
    end
end
disp('Comparison of Test Set Spectrum-Level Performance for MRMRLDA with different outlier strategies:');
disp(comp_table_data);

% Also compare probe-level performance (e.g., F2 score)
comp_table_data_probe = cell(length(metricNames)+1, length(outlierStrategies)+1);
comp_table_data_probe(1,1) = {'Metric (Probe Level)'};
for i=1:length(outlierStrategies), comp_table_data_probe(1,i+1) = {sprintf('TestSet Perf (%s)', outlierStrategies{i})}; end
for m=1:length(metricNames)
    comp_table_data_probe(m+1,1) = metricNames(m);
    for s=1:length(outlierStrategies)
        strat_name_field = sprintf('Strategy_%s', outlierStrategies{s});
        if isfield(overallPhase3Results, strat_name_field) && ...
           isfield(overallPhase3Results.(strat_name_field), 'probeLevelPerformance_MeanProb') && ...
           isfield(overallPhase3Results.(strat_name_field).probeLevelPerformance_MeanProb, metricNames{m})
            
            val_probe = overallPhase3Results.(strat_name_field).probeLevelPerformance_MeanProb.(metricNames{m});
            comp_table_data_probe(m+1,s+1) = {sprintf('%.4f', val_probe)};
        else
            comp_table_data_probe(m+1,s+1) = {'N/A'};
        end
    end
end
disp(char(10)); % New line
disp('Comparison of Test Set Probe-Level Performance (MeanProb) for MRMRLDA with different outlier strategies:');
disp(comp_table_data_probe);

% Save the overall results structure which contains performance for both
overallResultsFilename = fullfile(resultsPath_P3, sprintf('%s_Phase3_ComparativeStrategies_OverallResults.mat', dateStr));
save(overallResultsFilename, 'overallPhase3Results', 'outlierStrategies', 'metricNames');
fprintf('\nOverall Phase 3 comparative results saved to: %s\n', overallResultsFilename);


fprintf('\nPHASE 3 Processing Complete (Comparative Strategies): %s\n', string(datetime('now')));