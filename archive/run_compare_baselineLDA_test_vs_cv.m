% run_compare_baselineLDA_test_vs_cv.m
%
% Script to:
% 1. Train the BaselineLDA pipeline on the entire training set using a
%    chosen "final" hyperparameter (binningFactor) from Phase 2.
% 2. Evaluate this model on the unseen test set.
% 3. Compare its test set performance to its nested CV performance from Phase 2.
%
% Date: 2025-05-15

%% 0. Initialization
% =========================================================================
clear; clc; close all;
fprintf('COMPARING BaselineLDA: Test Set vs. Nested CV Performance - %s\n', string(datetime('now')));

% --- Define Paths ---
P = setup_project_paths();

addpath(P.helperFunPath);

dataPath    = P.dataPath;
resultsPath = P.resultsPath; % For loading Phase 2 results

dateStr = string(datetime('now','Format','yyyyMMdd')); % For any new outputs if needed

%% 1. Load Data & Phase 2 Results
% =========================================================================
% --- Load Entire Training Set ---
fprintf('Loading entire training set...\n');
trainingDataFile = fullfile(dataPath, 'training_set_no_outliers_T2Q.mat');
try
    loadedTrainingData = load(trainingDataFile, ...
                       'X_train_no_outliers', 'y_train_numeric_no_outliers');
    X_train_full = loadedTrainingData.X_train_no_outliers;
    y_train_full = loadedTrainingData.y_train_numeric_no_outliers;

    wavenumbers_data = load(fullfile(dataPath, 'wavenumbers.mat'), 'wavenumbers_roi');
    wavenumbers_original = wavenumbers_data.wavenumbers_roi;
    fprintf('Training data loaded: %d spectra, %d features.\n', size(X_train_full, 1), size(X_train_full, 2));
catch ME
    fprintf('ERROR loading training data from %s: %s\n', trainingDataFile, ME.message);
    return;
end

% --- Load Test Set ---
fprintf('Loading test set...\n');
testDataFile = fullfile(dataPath, 'data_table_test.mat');
try
    loadedTestData = load(testDataFile, 'dataTableTest');
    dataTableTest = loadedTestData.dataTableTest;
    numTestProbes = height(dataTableTest);
    temp_X_test_list = cell(numTestProbes, 1);
    temp_y_test_list = cell(numTestProbes, 1);
    for i = 1:numTestProbes
        spectraMatrix = dataTableTest.CombinedSpectra{i,1};
        numSpectraThisProbe = size(spectraMatrix, 1);
        temp_X_test_list{i} = spectraMatrix;
        current_WHO_grade_cat = dataTableTest.WHO_Grade(i);
        if current_WHO_grade_cat == 'WHO-1', temp_y_test_list{i} = ones(numSpectraThisProbe, 1) * 1;
        elseif current_WHO_grade_cat == 'WHO-3', temp_y_test_list{i} = ones(numSpectraThisProbe, 1) * 3;
        else, temp_y_test_list{i} = ones(numSpectraThisProbe, 1) * NaN; end
    end
    X_test_full = vertcat(temp_X_test_list{:});
    y_test_full_numeric = vertcat(temp_y_test_list{:});
    nan_label_idx_test = isnan(y_test_full_numeric);
    if any(nan_label_idx_test)
        X_test_full(nan_label_idx_test,:) = [];
        y_test_full_numeric(nan_label_idx_test) = [];
    end
    fprintf('Test data loaded and processed: %d spectra, %d features from %d probes.\n', ...
            size(X_test_full, 1), size(X_test_full, 2), numTestProbes);
catch ME
    fprintf('ERROR loading or processing test data from %s: %s\n', testDataFile, ME.message); return;
end

% --- Load Phase 2 Results to get BaselineLDA CV performance ---
fprintf('Loading Phase 2 results...\n');
% Find the latest Phase 2 results file (or specify exact filename)
phase2ResultsDir = fullfile(resultsPath, 'Phase2');
resultFiles = dir(fullfile(phase2ResultsDir, '*_Phase2_AllPipelineResults.mat'));
if isempty(resultFiles)
    error('No Phase 2 results file found in %s. Run Phase 2 script first.', phase2ResultsDir);
end
% Assuming the latest file is the one to use
[~,idxSort] = sort([resultFiles.datenum],'descend');
latestResultFile = fullfile(phase2ResultsDir, resultFiles(idxSort(1)).name);
fprintf('Using Phase 2 results from: %s\n', latestResultFile);
try
    phase2Data = load(latestResultFile, 'allPipelinesResults', 'pipelines', 'metricNames');
    allPipelinesResults_p2 = phase2Data.allPipelinesResults;
    pipelines_p2 = phase2Data.pipelines; % This is a cell array of structs
    metricNames_p2 = phase2Data.metricNames;
catch ME
    fprintf('ERROR loading Phase 2 results from %s: %s\n', latestResultFile, ME.message); return;
end

% Find BaselineLDA results from Phase 2
baselineLDACVResults = [];
for i = 1:length(pipelines_p2)
    if strcmpi(pipelines_p2{i}.name, 'BaselineLDA') % Use {} for cell array
        baselineLDACVResults = allPipelinesResults_p2{i};
        break;
    end
end
if isempty(baselineLDACVResults)
    error('BaselineLDA results not found in the loaded Phase 2 results file.');
end
fprintf('BaselineLDA Nested CV (Phase 2) Mean Performance (F2_WHO3): %.4f\n', ...
    baselineLDACVResults.outerFoldMetrics_mean(strcmpi(metricNames_p2, 'F2_WHO3')));

%% 2. Define Final Hyperparameter for BaselineLDA
% =========================================================================
% Based on Phase 2 results (e.g., mode or specific choice from outerFoldBestHyperparams for BaselineLDA)
% From your last output, binningFactors were [2, 2, 8, 8, 4] for BaselineLDA outer folds.
% Let's choose the mode, or one of the modes. We'll pick 2.
final_binningFactor_baseline = 2; % YOU CAN CHANGE THIS if you prefer 8 or 4
fprintf('Chosen final hyperparameter for BaselineLDA: Binning Factor = %d\n', final_binningFactor_baseline);

%% 3. Train Final BaselineLDA Model on Entire Training Set
% =========================================================================
fprintf('\n--- Training Final BaselineLDA Model on Entire Training Set ---\n');

% --- 3.1. Apply Binning to Full Training Set ---
fprintf('Applying binning (Factor: %d) to full training set...\n', final_binningFactor_baseline);
if final_binningFactor_baseline > 1
    [X_train_binned_baseline, ~] = bin_spectra(X_train_full, wavenumbers_original, final_binningFactor_baseline);
else
    X_train_binned_baseline = X_train_full;
end
fprintf('Training data after binning: %d spectra, %d features.\n', size(X_train_binned_baseline,1), size(X_train_binned_baseline,2));

% --- 3.2. Train Final LDA Classifier ---
fprintf('Training final LDA model for BaselineLDA pipeline...\n');
if isempty(X_train_binned_baseline) || size(X_train_binned_baseline,1) < 2 || length(unique(y_train_full)) < 2
    error('Insufficient data for final LDA training (BaselineLDA).');
end
final_BaselineLDA_model = fitcdiscr(X_train_binned_baseline, y_train_full);
fprintf('Final BaselineLDA model trained.\n');

%% 4. Evaluate Final BaselineLDA Model on Test Set
% =========================================================================
fprintf('\n--- Evaluating Final BaselineLDA Model on Unseen Test Set ---\n');

% --- 4.1. Apply Binning to Test Set ---
fprintf('Applying binning (Factor: %d) to test set...\n', final_binningFactor_baseline);
if final_binningFactor_baseline > 1
    [X_test_binned_baseline, ~] = bin_spectra(X_test_full, wavenumbers_original, final_binningFactor_baseline);
else
    X_test_binned_baseline = X_test_full;
end
fprintf('Test data after binning: %d spectra, %d features.\n', size(X_test_binned_baseline,1), size(X_test_binned_baseline,2));

% --- 4.2. Predict on Test Set ---
fprintf('Predicting on test set with BaselineLDA model...\n');
[y_pred_test_baseline, y_scores_test_baseline] = predict(final_BaselineLDA_model, X_test_binned_baseline);

% --- 4.3. Calculate Performance Metrics on Test Set ---
fprintf('Calculating performance metrics on test set for BaselineLDA...\n');
positiveClassLabel_test = 3; 
classOrder_test = final_BaselineLDA_model.ClassNames;
positiveClassColIdx_test = [];
if isnumeric(classOrder_test) && isnumeric(positiveClassLabel_test)
     positiveClassColIdx_test = find(classOrder_test == positiveClassLabel_test);
elseif iscategorical(classOrder_test) && isnumeric(positiveClassLabel_test)
     positiveClassColIdx_test = find(str2double(string(classOrder_test)) == positiveClassLabel_test);
elseif iscellstr(classOrder_test) && isnumeric(positiveClassLabel_test)
     positiveClassColIdx_test = find(str2double(classOrder_test) == positiveClassLabel_test);
else 
     positiveClassColIdx_test = find(classOrder_test == positiveClassLabel_test);
end

if isempty(positiveClassColIdx_test) || ( ~isempty(positiveClassColIdx_test) && (max(positiveClassColIdx_test) > size(y_scores_test_baseline,2)) )
    error('Positive class label %d not found or scores issue in final BaselineLDA model for test set evaluation.', positiveClassLabel_test);
end
scores_for_positive_class_test_baseline = y_scores_test_baseline(:, positiveClassColIdx_test);

% Define metricNames for this script if not loaded from Phase 2 results, or ensure consistency
if ~exist('metricNames_p2', 'var') || isempty(metricNames_p2)
    metricNames_compare = {'Accuracy', 'Sensitivity_WHO3', 'Specificity_WHO1', 'PPV_WHO3', 'NPV_WHO1', 'F1_WHO3', 'F2_WHO3', 'AUC'};
else
    metricNames_compare = metricNames_p2;
end

baselineLDATestSetPerformance = calculate_performance_metrics(y_test_full_numeric, y_pred_test_baseline, scores_for_positive_class_test_baseline, positiveClassLabel_test, metricNames_compare);

%% 5. Display Comparison
% =========================================================================
fprintf('\n\n--- Comparison: BaselineLDA Performance ---\n');
fprintf('Chosen Binning Factor for this Test: %d\n\n', final_binningFactor_baseline);

comparison_data = cell(length(metricNames_compare) + 1, 3);
comparison_data(1,:) = {'Metric', 'Nested CV Mean (Phase 2)', 'Test Set Performance'};

for i = 1:length(metricNames_compare)
    metric_name = metricNames_compare{i};
    cv_mean_val = baselineLDACVResults.outerFoldMetrics_mean(strcmpi(metricNames_p2, metric_name));
    test_val = baselineLDATestSetPerformance.(metric_name);
    comparison_data(i+1,:) = {metric_name, sprintf('%.4f', cv_mean_val), sprintf('%.4f', test_val)};
end

disp(comparison_data);

% Convert to table for nicer display if preferred
comparison_table = cell2table(comparison_data(2:end,:), 'VariableNames', comparison_data(1,:));
fprintf('\nFormatted Table:\n');
disp(comparison_table);


fprintf('\nBaselineLDA Comparison Script Complete: %s\n', string(datetime('now')));