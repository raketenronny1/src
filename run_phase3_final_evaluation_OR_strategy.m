function run_phase3_final_evaluation_OR_strategy(cfg)
%RUN_PHASE3_FINAL_EVALUATION_OR_STRATEGY
%
% Script for Phase 3: Final Model Training & Unbiased Evaluation.
% Focuses on the "T2 OR Q" outlier strategy.
% 1. Loads the "T2 OR Q" cleaned training set.
% 2. Loads best hyperparameters for MRMRLDA (for OR strategy) from Phase 2.
% 3. Trains the MRMRLDA pipeline on the entire "T2 OR Q" training set.
% 4. Evaluates the final model on the common unseen test set.
% 5. Saves results and models.
%
% Date: 2025-05-18 (Simplified to focus on OR strategy)

%% 0. Initialization
% =========================================================================
fprintf('PHASE 3: Final Model Training & Unbiased Evaluation (T2 OR Q Strategy) - %s\n', string(datetime('now')));

if nargin < 1
    cfg = struct();
end
if ~isfield(cfg, 'projectRoot')
    cfg.projectRoot = pwd;
end

% --- Define Paths ---
P = setup_project_paths(cfg.projectRoot, 'Phase3');
dataPath         = P.dataPath;
phase2ModelsPath = fullfile(P.projectRoot, 'models', 'Phase2'); % For loading best hyperparameters
resultsPath_P3   = P.resultsPath; % General Phase 3 results
modelsPath_P3    = P.modelsPath;
figuresPath_P3   = P.figuresPath;

if ~exist(resultsPath_P3, 'dir'), mkdir(resultsPath_P3); end
if ~exist(modelsPath_P3, 'dir'), mkdir(modelsPath_P3); end
if ~exist(figuresPath_P3, 'dir'), mkdir(figuresPath_P3); end

dateStr = string(datetime('now','Format','yyyyMMdd'));

% Colors for plots
colorWHO1 = [0.9, 0.6, 0.4]; % Orange
colorWHO3 = [0.4, 0.702, 0.902]; % Blue

currentStrategy = 'OR'; % Hardcoded to OR strategy

%% 1. Load Data & Best Pipeline Info for OR Strategy
% =========================================================================

% --- Load Wavenumbers (common for all) ---
try
    wavenumbers_data = load(fullfile(dataPath, 'wavenumbers.mat'), 'wavenumbers_roi');
    wavenumbers_original = wavenumbers_data.wavenumbers_roi;
    if iscolumn(wavenumbers_original), wavenumbers_original = wavenumbers_original'; end
catch ME_wave
    error('ERROR loading wavenumbers.mat: %s\n', ME_wave.message);
end

% --- Load "T2 OR Q" Cleaned Training Set ---
fprintf('Loading "T2 OR Q" cleaned training set...\n');
trainingDataFilePattern_OR = '*_training_set_no_outliers_T2orQ.mat';
varName_X_OR = 'X_train_no_outliers_OR';
varName_y_OR = 'y_train_no_outliers_OR_num'; 
varName_probeIDs_OR = 'Patient_ID_no_outliers_OR'; % <<< CORRECTED VARIABLE NAME TO LOAD

trainingDataFiles_OR = dir(fullfile(dataPath, trainingDataFilePattern_OR));
if isempty(trainingDataFiles_OR)
    error('No training set file found for "OR" strategy in %s (pattern: %s).', dataPath, trainingDataFilePattern_OR);
end
[~,idxSortTrain_OR] = sort([trainingDataFiles_OR.datenum],'descend');
latestTrainingDataFile_OR = fullfile(dataPath, trainingDataFiles_OR(idxSortTrain_OR(1)).name);

try
    % Load all necessary variables explicitly
    loadedTrainingData_OR = load(latestTrainingDataFile_OR, varName_X_OR, varName_y_OR, varName_probeIDs_OR); 
    
    X_train_full = loadedTrainingData_OR.(varName_X_OR);
    y_train_full = loadedTrainingData_OR.(varName_y_OR);
    probeIDs_train_full = loadedTrainingData_OR.(varName_probeIDs_OR); % <<< USE CORRECTED VARIABLE NAME

    if ~isnumeric(y_train_full), error('y_train_full for OR strategy is not numeric.'); end
    y_train_full = y_train_full(:);

    fprintf('Training data for OR strategy loaded: %d spectra, %d features from %s.\n', ...
        size(X_train_full, 1), size(X_train_full, 2), latestTrainingDataFile_OR);
    if isempty(X_train_full), error('X_train_full is empty for OR strategy.'); end
catch ME_load_train_OR
    fprintf('ERROR loading OR strategy training data from %s: %s\n', latestTrainingDataFile_OR, ME_load_train_OR.message);
    % Check if the error is due to a missing variable specifically
    if contains(ME_load_train_OR.message, 'Variable') && contains(ME_load_train_OR.message, 'not found')
        fprintf('Please ensure the .mat file "%s" contains the variables: %s, %s, and %s.\n', latestTrainingDataFile_OR, varName_X_OR, varName_y_OR, varName_probeIDs_OR);
    end
    rethrow(ME_load_train_OR);
end


% --- Load Test Set (Common) ---
fprintf('Loading common test set...\n');
testDataFile = fullfile(dataPath, 'data_table_test.mat'); 
try
    loadedTestData = load(testDataFile, 'dataTableTest');
    dataTableTest = loadedTestData.dataTableTest;

    numTestProbes = height(dataTableTest);
    temp_X_test_list = cell(numTestProbes, 1);
    temp_y_test_list = cell(numTestProbes, 1);
    temp_probeIDs_test_list = cell(numTestProbes, 1);
    
    for i_test = 1:numTestProbes
        spectraMatrix_test = dataTableTest.CombinedSpectra{i_test,1}; 
        numSpectraThisProbe_test = size(spectraMatrix_test, 1);
        temp_X_test_list{i_test} = spectraMatrix_test;
        current_WHO_grade_cat_test = dataTableTest.WHO_Grade(i_test);
        if current_WHO_grade_cat_test == 'WHO-1', temp_y_test_list{i_test} = ones(numSpectraThisProbe_test, 1) * 1;
        elseif current_WHO_grade_cat_test == 'WHO-3', temp_y_test_list{i_test} = ones(numSpectraThisProbe_test, 1) * 3;
        else, temp_y_test_list{i_test} = ones(numSpectraThisProbe_test, 1) * NaN; end
        temp_probeIDs_test_list{i_test} = repmat(dataTableTest.Diss_ID(i_test), numSpectraThisProbe_test, 1);
    end
    
    X_test_full = vertcat(temp_X_test_list{:});
    y_test_full_numeric = vertcat(temp_y_test_list{:});
    y_test_full_numeric = y_test_full_numeric(:); 
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
    if ~isempty(X_train_full) && size(X_test_full,2) ~= size(X_train_full,2)
        error('Mismatch in number of features between training (%d) and test (%d) data before binning.', size(X_train_full,2), size(X_test_full,2));
    end
catch ME_load_test
    fprintf('ERROR loading or processing common test data from %s: %s\n', testDataFile, ME_load_test.message);
    rethrow(ME_load_test);
end

% --- Load Final Model Hyperparameters for MRMRLDA for "OR" Strategy ---
fprintf('Loading best hyperparameters for MRMRLDA (Strategy: OR) from Phase 2 model info...\n');
bestHyperparamFilePattern_OR = fullfile(phase2ModelsPath, sprintf('*_Phase2_BestPipelineInfo_Strat_OR.mat'));
bestHyperparamFiles_OR = dir(bestHyperparamFilePattern_OR);


% Default hyperparameters if Phase 2 results are unavailable
final_binningFactor = 1; % Adjust if Phase 2 recommended a different factor
final_mrmrFeaturePercent = 0.1; % Select 10%% of features

if isempty(bestHyperparamFiles_OR)
    warning('No Phase 2 best hyperparameter file found in %s for MRMRLDA and strategy OR using pattern "%s". Using predefined defaults (Binning: %d, MRMR Percent: %.2f).', ...
        phase2ModelsPath, sprintf('*_Phase2_BestPipelineInfo_Strat_OR.mat', final_binningFactor, final_mrmrFeaturePercent));
else
    [~,idxSortHP_OR] = sort([bestHyperparamFiles_OR.datenum],'descend');
    latestBestHPFile_OR = fullfile(phase2ModelsPath, bestHyperparamFiles_OR(idxSortHP_OR(1)).name);
    fprintf('Loading best hyperparameters for OR strategy from: %s\n', latestBestHPFile_OR);
    try
        loadedHPData_OR = load(latestBestHPFile_OR, 'bestPipelineSummary_strat');
        bestPipelineSummary_OR = loadedHPData_OR.bestPipelineSummary_strat;

        if strcmpi(bestPipelineSummary_OR.pipelineConfig.name, 'MRMRLDA')
            all_binning_factors_p2_OR = [];
            all_mrmr_features_p2_OR = [];
            if isfield(bestPipelineSummary_OR, 'outerFoldBestHyperparams') && iscell(bestPipelineSummary_OR.outerFoldBestHyperparams)
                for k_hp = 1:length(bestPipelineSummary_OR.outerFoldBestHyperparams)
                    if ~isempty(bestPipelineSummary_OR.outerFoldBestHyperparams{k_hp})
                        if isfield(bestPipelineSummary_OR.outerFoldBestHyperparams{k_hp}, 'binningFactor')
                            all_binning_factors_p2_OR = [all_binning_factors_p2_OR; bestPipelineSummary_OR.outerFoldBestHyperparams{k_hp}.binningFactor];
                        end
                        if isfield(bestPipelineSummary_OR.outerFoldBestHyperparams{k_hp}, 'mrmrFeaturePercent')
                            all_mrmr_features_p2_OR = [all_mrmr_features_p2_OR; bestPipelineSummary_OR.outerFoldBestHyperparams{k_hp}.mrmrFeaturePercent];
                        end
                    end
                end
            end
            
            if isempty(all_binning_factors_p2_OR)
                warning('No binning factors found in loaded bestPipelineSummary_strat for MRMRLDA (OR strategy). Using fixed default %d.', final_binningFactor);
            end
            if ~isempty(all_mrmr_features_p2_OR)
                final_mrmrFeaturePercent = mode(all_mrmr_features_p2_OR);
            else
                warning('No mrmrFeaturePercent found in loaded bestPipelineSummary_strat for MRMRLDA (OR strategy). Using predefined default %.2f.', final_mrmrFeaturePercent);
            end
            fprintf('Using hyperparameters for MRMRLDA (OR Strategy) from Phase 2: Binning=%d, MRMR Percent=%.2f\n', ...
                     final_binningFactor, final_mrmrFeaturePercent);
        else
            warning('The best pipeline found in Phase 2 for OR strategy was %s, not MRMRLDA. Using predefined defaults for MRMRLDA (Binning: %d, MRMR Percent: %.2f).', ...
                    bestPipelineSummary_OR.pipelineConfig.name, final_binningFactor, final_mrmrFeaturePercent);
        end
    catch ME_loadHP_OR
        fprintf('ERROR loading/processing best hyperparameters for OR strategy from %s: %s. Using predefined defaults (Binning: %d, MRMR Percent: %.2f).\n', ...
            latestBestHPFile_OR, ME_loadHP_OR.message, final_binningFactor, final_mrmrFeaturePercent);
    end
end

% --- Define Metric Names ---
metricNames = {'Accuracy', 'Sensitivity_WHO3', 'Specificity_WHO1', 'PPV_WHO3', 'NPV_WHO1', 'F1_WHO3', 'F2_WHO3', 'AUC'};

% --- FROM HERE ON, THE SCRIPT IS VERY SIMILAR TO THE ORIGINAL run_phase3_final_evaluation.m ---
% --- It uses X_train_full, y_train_full specific to the OR strategy, and the determined hyperparameters ---

%% 2. Train Final Model on Entire Training Set (MRMRLDA with OR strategy data)
% =========================================================================
fprintf('\n--- Training Final MRMRLDA Model on Entire Training Set (Strategy: %s) ---\n', currentStrategy); % currentStrategy is 'OR'

% Apply Binning
if final_binningFactor > 1
    [X_train_binned, wavenumbers_binned] = bin_spectra(X_train_full, wavenumbers_original, final_binningFactor);
else
    X_train_binned = X_train_full;
    wavenumbers_binned = wavenumbers_original;
    fprintf('No binning applied. Using original training features.\n');
end
fprintf('Training data for OR strategy after binning: %d spectra, %d features.\n', size(X_train_binned,1), size(X_train_binned,2));

% Apply MRMR
y_train_cat = categorical(y_train_full); 
final_selected_feature_indices_in_binned_space = [];
final_selected_wavenumbers = [];
X_train_fs = X_train_binned; % Default if no MRMR features or selection fails

final_numMRMRFeatures = ceil(final_mrmrFeaturePercent * size(X_train_binned,2));
if final_numMRMRFeatures > 0 && size(X_train_binned, 2) > 0 && final_numMRMRFeatures <= size(X_train_binned,2)
    try
        [ranked_indices_all_train, ~] = fscmrmr(X_train_binned, y_train_cat);
        actual_num_to_select_final = min(final_numMRMRFeatures, length(ranked_indices_all_train));
        if actual_num_to_select_final < final_numMRMRFeatures
            warning('MRMR: Requested %d features, but only %d available/selected from OR training set. Using %d.', ...
                    final_numMRMRFeatures, length(ranked_indices_all_train), actual_num_to_select_final);
        end
        if actual_num_to_select_final > 0
            final_selected_feature_indices_in_binned_space = ranked_indices_all_train(1:actual_num_to_select_final);
            final_selected_wavenumbers = wavenumbers_binned(final_selected_feature_indices_in_binned_space);
            X_train_fs = X_train_binned(:, final_selected_feature_indices_in_binned_space);
        else
             warning('MRMR resulted in 0 features for OR strategy. Using all binned features for LDA.');
        end
    catch ME_mrmr_final
        fprintf('ERROR during final MRMR for OR strategy: %s. Using all binned features.\n', ME_mrmr_final.message);
    end
elseif final_numMRMRFeatures > size(X_train_binned,2)
    warning('Requested numMRMRFeatures (%d) > available binned features (%d) for OR strategy. Using all binned features.', ...
        final_numMRMRFeatures, size(X_train_binned,2));
end

fprintf('MRMR selected %d features. Final training data for LDA (OR Strategy): %d spectra, %d features.\n', ...
    length(final_selected_feature_indices_in_binned_space), size(X_train_fs,1), size(X_train_fs,2));

% Train LDA
if isempty(X_train_fs) || size(X_train_fs,1) < 2 || length(unique(y_train_full)) < 2
    error('Insufficient data for final LDA training for OR strategy.');
end
final_LDAModel = fitcdiscr(X_train_fs, y_train_full); % Model is now just final_LDAModel
fprintf('Final LDA model trained for OR strategy.\n');

%% 3. Evaluate Final Model on Test Set
% =========================================================================
fprintf('\n--- Evaluating Final Model on Unseen Test Set (using OR Strategy Model) ---\n');

% Apply Binning to Test Set
if final_binningFactor > 1
    [X_test_binned, ~] = bin_spectra(X_test_full, wavenumbers_original, final_binningFactor);
else
    X_test_binned = X_test_full;
    fprintf('No binning applied to test set. Using original features.\n');
end
fprintf('Test data after binning: %d spectra, %d features.\n', size(X_test_binned,1), size(X_test_binned,2));

% Apply Feature Selection to Test Set (using indices from training)
X_test_fs = X_test_binned; 
if ~isempty(final_selected_feature_indices_in_binned_space)
    X_test_fs = X_test_binned(:, final_selected_feature_indices_in_binned_space);
end
fprintf('Test data for LDA: %d spectra, %d features.\n', size(X_test_fs,1), size(X_test_fs,2));

% Predict on Test Set
fprintf('Predicting on test set with OR strategy model...\n');
[y_pred_test, y_scores_test] = predict(final_LDAModel, X_test_fs); % Using final_LDAModel

% Calculate Spectrum-Level Performance Metrics
fprintf('Calculating performance metrics on test set for OR strategy model...\n');
positiveClassLabel = 3; 
classOrder = final_LDAModel.ClassNames;
positiveClassColIdx = find(classOrder == positiveClassLabel); 
if isempty(positiveClassColIdx) || max(positiveClassColIdx) > size(y_scores_test,2)
    error('Positive class for AUC not found correctly for test set.');
end
scores_for_positive_class_test = y_scores_test(:, positiveClassColIdx);
testSetPerformanceMetrics_spectrum = calculate_performance_metrics(y_test_full_numeric, y_pred_test, scores_for_positive_class_test, positiveClassLabel, metricNames);
fprintf('\n--- Test Set Performance Metrics (Spectrum-Level, OR Strategy Model) ---\n');
disp(struct2table(testSetPerformanceMetrics_spectrum));

%% 4. Probe-Level Aggregation and Evaluation
% =========================================================================
fprintf('\n--- Aggregating results to Probe-Level for Test Set (OR Strategy Model) ---\n');
uniqueTestProbes = unique(probeIDs_test_full, 'stable');
numUniqueTestProbes = length(uniqueTestProbes);
probeLevelResults = table('Size', [numUniqueTestProbes, 7], ...
    'VariableTypes', {'cell', 'double', 'categorical', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'Diss_ID', 'True_WHO_Grade_Numeric', 'True_WHO_Grade_Category', ...
                      'Mean_WHO3_Probability', 'Predicted_WHO_Grade_Numeric_MeanProb', ...
                      'MajorityVote_Predicted_WHO_Grade_Numeric', 'Proportion_Spectra_Predicted_WHO3'});
probeLevelResults.Diss_ID = uniqueTestProbes;

for i_probe = 1:numUniqueTestProbes
    currentProbeID = uniqueTestProbes{i_probe};
    idxSpectraForProbe = strcmp(probeIDs_test_full, currentProbeID);
    
    true_labels_this_probe = y_test_full_numeric(idxSpectraForProbe);
    if ~isempty(true_labels_this_probe)
        probeLevelResults.True_WHO_Grade_Numeric(i_probe) = mode(true_labels_this_probe);
        if mode(true_labels_this_probe) == 1, probeLevelResults.True_WHO_Grade_Category(i_probe) = 'WHO-1';
        elseif mode(true_labels_this_probe) == 3, probeLevelResults.True_WHO_Grade_Category(i_probe) = 'WHO-3'; end
    end
    
    mean_prob_who3 = mean(scores_for_positive_class_test(idxSpectraForProbe));
    probeLevelResults.Mean_WHO3_Probability(i_probe) = mean_prob_who3;
    otherClasses = setdiff(unique(y_train_full), positiveClassLabel); 
    defaultOtherClass = 1; if ~isempty(otherClasses), defaultOtherClass = otherClasses(1); end
    if mean_prob_who3 > 0.5, probeLevelResults.Predicted_WHO_Grade_Numeric_MeanProb(i_probe) = positiveClassLabel;
    else, probeLevelResults.Predicted_WHO_Grade_Numeric_MeanProb(i_probe) = defaultOtherClass; end
    
    predicted_labels_this_probe = y_pred_test(idxSpectraForProbe);
    probeLevelResults.MajorityVote_Predicted_WHO_Grade_Numeric(i_probe) = mode(predicted_labels_this_probe);
    probeLevelResults.Proportion_Spectra_Predicted_WHO3(i_probe) = sum(predicted_labels_this_probe == positiveClassLabel) / length(predicted_labels_this_probe);
end
fprintf('\nProbe-Level Aggregated Results (Test Set, OR Strategy Model):\n');
disp(probeLevelResults);

% Calculate Probe-Level Performance Metrics
y_true_probe = probeLevelResults.True_WHO_Grade_Numeric;
y_pred_probe_mean_prob = probeLevelResults.Predicted_WHO_Grade_Numeric_MeanProb;
probe_scores_for_positive_class = probeLevelResults.Mean_WHO3_Probability;
valid_probes_idx = ~isnan(y_true_probe) & ~isnan(y_pred_probe_mean_prob);

if sum(valid_probes_idx) > 0 && length(unique(y_true_probe(valid_probes_idx))) > 1
    fprintf('\n--- Probe-Level Performance Metrics (based on Mean Probability > 0.5, OR Strategy Model) ---\n');
    probeLevelPerfMetrics = calculate_performance_metrics(y_true_probe(valid_probes_idx), ...
        y_pred_probe_mean_prob(valid_probes_idx), probe_scores_for_positive_class(valid_probes_idx), positiveClassLabel, metricNames);
    disp(struct2table(probeLevelPerfMetrics));
else
    fprintf('\nCould not calculate probe-level performance metrics for OR strategy (e.g., only one class present or no valid probes).\n');
    probeLevelPerfMetrics = struct(); 
    for m=1:length(metricNames), probeLevelPerfMetrics.(metricNames{m}) = NaN; end
end

%% 5. Save Final Model and Results (OR Strategy)
% =========================================================================
finalModelPackage_OR = struct(); % Name clearly indicates OR strategy
finalModelPackage_OR.description = 'Final MRMRLDA model (OR Strategy) trained on the T2orQ cleaned training set.';
finalModelPackage_OR.trainingDate = string(datetime('now'));
finalModelPackage_OR.LDAModel = final_LDAModel; % The model trained with OR data
finalModelPackage_OR.binningFactor = final_binningFactor;
finalModelPackage_OR.numMRMRFeaturesSelected = length(final_selected_feature_indices_in_binned_space);
finalModelPackage_OR.mrmrFeaturePercent = final_mrmrFeaturePercent;
finalModelPackage_OR.selectedFeatureIndices_in_binned_space = final_selected_feature_indices_in_binned_space;
finalModelPackage_OR.selectedWavenumbers = final_selected_wavenumbers;
finalModelPackage_OR.originalWavenumbers_before_binning = wavenumbers_original;
finalModelPackage_OR.binnedWavenumbers_for_selection = wavenumbers_binned; 
finalModelPackage_OR.testSetPerformance_Spectrum = testSetPerformanceMetrics_spectrum;
finalModelPackage_OR.probeLevelResults = probeLevelResults;
finalModelPackage_OR.probeLevelPerformance_MeanProb = probeLevelPerfMetrics;
finalModelPackage_OR.trainingDataFile = latestTrainingDataFile_OR; 
finalModelPackage_OR.testDataFile = testDataFile;
finalModelPackage_OR.outlierStrategyUsed = 'OR';

% Save to general Phase 3 models/results folder, clearly named
modelFilename_OR = fullfile(modelsPath_P3, sprintf('%s_Phase3_FinalMRMRLDA_Model_Strat_OR.mat', dateStr));
save(modelFilename_OR, 'finalModelPackage_OR'); 
fprintf('\nFinal model package for OR strategy saved to: %s\n', modelFilename_OR);

resultsFilename_phase3_OR = fullfile(resultsPath_P3, sprintf('%s_Phase3_TestSetResults_Strat_OR.mat', dateStr));
save(resultsFilename_phase3_OR, 'testSetPerformanceMetrics_spectrum', 'probeLevelResults', 'probeLevelPerfMetrics', ...
     'final_binningFactor', 'final_numMRMRFeatures', 'final_mrmrFeaturePercent', 'final_selected_wavenumbers');
fprintf('Phase 3 test set results for OR strategy saved to: %s\n', resultsFilename_phase3_OR);

%% 6. Visualizations (OR Strategy)
% =========================================================================
% Spectrum-Level Confusion Matrix
if exist('confusionchart','file') && ~isempty(y_test_full_numeric) && ~isempty(y_pred_test)
    figConfSpec_OR = figure('Name', 'Confusion Matrix (Spectrum, Test Set) - OR Strategy');
    cm_spec_OR = confusionchart(y_test_full_numeric, y_pred_test, ...
        'ColumnSummary','column-normalized', 'RowSummary','row-normalized', ...
        'Title', sprintf('CM Spectrum (Test) - Strat: OR (F2: %.3f)', testSetPerformanceMetrics_spectrum.F2_WHO3));
    confMatFigFilenameBase_spec_OR = fullfile(figuresPath_P3, sprintf('%s_Phase3_ConfMat_Spectrum_Strat_OR', dateStr));
    savefig(figConfSpec_OR, [confMatFigFilenameBase_spec_OR, '.fig']);
    exportgraphics(figConfSpec_OR, [confMatFigFilenameBase_spec_OR, '.tiff'], 'Resolution', 300);
    close(figConfSpec_OR);
end

% Probe-Level Confusion Matrix (Mean Probability)
if exist('confusionchart','file') && sum(valid_probes_idx) > 0 && length(unique(y_true_probe(valid_probes_idx))) > 1
    figConfProbe_OR = figure('Name', 'Confusion Matrix (Probe, Mean Prob, Test Set) - OR Strategy');
    cm_probe_OR = confusionchart(y_true_probe(valid_probes_idx), y_pred_probe_mean_prob(valid_probes_idx), ...
        'ColumnSummary','column-normalized', 'RowSummary','row-normalized', ...
        'Title', sprintf('CM Probe (MeanProb, Test) - Strat: OR (F2: %.3f)', probeLevelPerfMetrics.F2_WHO3));
    confMatProbeFigFilenameBase_OR = fullfile(figuresPath_P3, sprintf('%s_Phase3_ConfMat_Probe_MeanProb_Strat_OR', dateStr));
    savefig(figConfProbe_OR, [confMatProbeFigFilenameBase_OR, '.fig']);
    exportgraphics(figConfProbe_OR, [confMatProbeFigFilenameBase_OR, '.tiff'], 'Resolution', 300);
    close(figConfProbe_OR);
end

% Probe-Level Probability Plot
figProbDist_OR = figure('Name', 'Probe-Level Mean WHO-3 Probabilities (Test Set) - OR Strategy', 'Position', [100, 100, 900, 700]);
hold on; jitterAmount = 0.02; 
probes_true_who1_plot_OR = probeLevelResults(probeLevelResults.True_WHO_Grade_Numeric == 1, :);
probes_true_who3_plot_OR = probeLevelResults(probeLevelResults.True_WHO_Grade_Numeric == 3, :);
h_p3_1_OR = []; h_p3_3_OR = [];
if ~isempty(probes_true_who1_plot_OR), x_coords_who1_OR = 1 + (rand(height(probes_true_who1_plot_OR),1)-0.5)*jitterAmount*2; h_p3_1_OR = scatter(x_coords_who1_OR, probes_true_who1_plot_OR.Mean_WHO3_Probability,70,'o','MarkerEdgeColor','k','MarkerFaceColor',colorWHO1,'LineWidth',1,'DisplayName','True WHO-1 Probes'); end
if ~isempty(probes_true_who3_plot_OR), x_coords_who3_OR = 2 + (rand(height(probes_true_who3_plot_OR),1)-0.5)*jitterAmount*2; h_p3_3_OR = scatter(x_coords_who3_OR, probes_true_who3_plot_OR.Mean_WHO3_Probability,70,'s','MarkerEdgeColor','k','MarkerFaceColor',colorWHO3,'LineWidth',1,'DisplayName','True WHO-3 Probes'); end
plot([0.5 2.5], [0.5 0.5], 'k--', 'DisplayName', 'Decision Threshold (0.5)');
hold off; xticks([1 2]); xticklabels({'True WHO-1', 'True WHO-3'}); xlim([0.5 2.5]); ylim([0 1]);
ylabel('Mean Predicted Probability of WHO-3'); title('Probe-Level Classification Probabilities (Test Set) - OR Strategy'); grid on;
if ~isempty(h_p3_1_OR) || ~isempty(h_p3_3_OR), legend([h_p3_1_OR, h_p3_3_OR], 'Location', 'best'); end; set(gca, 'FontSize', 12);
probeProbFigFilenameBase_OR = fullfile(figuresPath_P3, sprintf('%s_Phase3_ProbeLevelProbabilities_Strat_OR', dateStr));
savefig(figProbDist_OR, [probeProbFigFilenameBase_OR, '.fig']);
exportgraphics(figProbDist_OR, [probeProbFigFilenameBase_OR, '.tiff'], 'Resolution', 300);
close(figProbDist_OR);

% Violin plots can also be generated here if desired, similar to the probe probability plot

fprintf('\nPHASE 3 Processing Complete (Focused on OR Strategy): %s\n', string(datetime('now')));
end
