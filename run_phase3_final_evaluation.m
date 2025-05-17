% run_phase3_final_evaluation.m
%
% Script for Phase 3: Final Model Training & Unbiased Evaluation.
% 1. Trains the best pipeline from Phase 2 (MRMRLDA) on the entire training set.
% 2. Evaluates the final model on the unseen test set.
%
% Date: 2025-05-15

%% 0. Initialization
% =========================================================================
clear; clc; close all;
fprintf('PHASE 3: Final Model Training & Unbiased Evaluation - %s\n', string(datetime('now')));

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
addpath(helperFunPath);

dataPath      = fullfile(projectRoot, 'data');
resultsPath   = fullfile(projectRoot, 'results', 'Phase3'); % Specific to Phase 3
modelsPath    = fullfile(projectRoot, 'models', 'Phase3');   % Specific to Phase 3
figuresPath   = fullfile(projectRoot, 'figures', 'Phase3'); % Specific to Phase 3

if ~exist(resultsPath, 'dir'), mkdir(resultsPath); end
if ~exist(modelsPath, 'dir'), mkdir(modelsPath); end
if ~exist(figuresPath, 'dir'), mkdir(figuresPath); end

dateStr = string(datetime('now','Format','yyyyMMdd'));

%% 1. Load Data & Best Pipeline Info
% =========================================================================
% --- Load Entire Training Set ---
fprintf('Loading entire training set...\n');
trainingDataFile = fullfile(dataPath, 'training_set_no_outliers_T2Q.mat');
try
    loadedTrainingData = load(trainingDataFile, ...
                       'X_train_no_outliers', 'y_train_numeric_no_outliers', ...
                       'patientIDs_train_no_outliers'); % Assuming these are the final training variables
    X_train_full = loadedTrainingData.X_train_no_outliers;
    y_train_full = loadedTrainingData.y_train_numeric_no_outliers;
    % probeIDs_train_full = loadedTrainingData.patientIDs_train_no_outliers; % May not be needed here unless for y_train_cat

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

    % Process dataTableTest to X_test_full, y_test_full_numeric, probeIDs_test_full
    numTestProbes = height(dataTableTest);
    temp_X_test_list = cell(numTestProbes, 1);
    temp_y_test_list = cell(numTestProbes, 1);
    temp_probeIDs_test_list = cell(numTestProbes, 1);
    
    totalTestSpectra = 0;
    for i = 1:numTestProbes
        spectraMatrix = dataTableTest.CombinedSpectra{i,1}; % This is N_spectra_for_probe_i x N_features
        numSpectraThisProbe = size(spectraMatrix, 1);
        totalTestSpectra = totalTestSpectra + numSpectraThisProbe;
        
        temp_X_test_list{i} = spectraMatrix;
        
        current_WHO_grade_cat = dataTableTest.WHO_Grade(i);
        if current_WHO_grade_cat == 'WHO-1'
            temp_y_test_list{i} = ones(numSpectraThisProbe, 1) * 1;
        elseif current_WHO_grade_cat == 'WHO-3'
            temp_y_test_list{i} = ones(numSpectraThisProbe, 1) * 3;
        else
            warning('Test probe %s has unexpected WHO grade: %s. Assigning NaN.', dataTableTest.Diss_ID{i}, string(current_WHO_grade_cat));
            temp_y_test_list{i} = ones(numSpectraThisProbe, 1) * NaN;
        end
        
        temp_probeIDs_test_list{i} = repmat(dataTableTest.Diss_ID(i), numSpectraThisProbe, 1);
    end
    
    X_test_full = vertcat(temp_X_test_list{:});
    y_test_full_numeric = vertcat(temp_y_test_list{:});
    probeIDs_test_full = vertcat(temp_probeIDs_test_list{:});

    % Remove any rows with NaN labels (if any unexpected WHO grades occurred)
    nan_label_idx_test = isnan(y_test_full_numeric);
    if any(nan_label_idx_test)
        fprintf('Removing %d test spectra with NaN labels due to unexpected WHO grades.\n', sum(nan_label_idx_test));
        X_test_full(nan_label_idx_test,:) = [];
        y_test_full_numeric(nan_label_idx_test) = [];
        probeIDs_test_full(nan_label_idx_test) = [];
    end

    fprintf('Test data loaded and processed: %d spectra, %d features from %d probes.\n', ...
            size(X_test_full, 1), size(X_test_full, 2), numTestProbes);

    if size(X_test_full,2) ~= size(X_train_full,2)
        error('Mismatch in number of features between training (%d) and test (%d) data before binning.', size(X_train_full,2), size(X_test_full,2));
    end

catch ME
    fprintf('ERROR loading or processing test data from %s: %s\n', testDataFile, ME.message);
    disp(ME.getReport);
    return;
end

% --- Define Final Model Hyperparameters (MRMRLDA) ---
% Based on Phase 2 results (mode of outerFoldBestHyperparams for MRMRLDA)
final_binningFactor = 8;
final_numMRMRFeatures = 50;
fprintf('Final hyperparameters for MRMRLDA: Binning Factor = %d, Num MRMR Features = %d\n', ...
    final_binningFactor, final_numMRMRFeatures);

% --- Define Metric Names (needed for calculate_performance_metrics) ---
metricNames = {'Accuracy', 'Sensitivity_WHO3', 'Specificity_WHO1', 'PPV_WHO3', 'NPV_WHO1', 'F1_WHO3', 'F2_WHO3', 'AUC'}; % <<<< ADD THIS LINE
%% 2. Train Final Model on Entire Training Set (MRMRLDA)
% =========================================================================
fprintf('\n--- Training Final MRMRLDA Model on Entire Training Set ---\n');

% --- 2.1. Apply Binning to Full Training Set ---
fprintf('Applying binning (Factor: %d) to full training set...\n', final_binningFactor);
if final_binningFactor > 1
    [X_train_binned, wavenumbers_binned] = bin_spectra(X_train_full, wavenumbers_original, final_binningFactor);
else
    X_train_binned = X_train_full;
    wavenumbers_binned = wavenumbers_original;
end
fprintf('Training data after binning: %d spectra, %d features.\n', size(X_train_binned,1), size(X_train_binned,2));

% --- 2.2. Apply MRMR Feature Selection to Full Binned Training Set ---
fprintf('Applying MRMR (Target Features: %d) to full binned training set...\n', final_numMRMRFeatures);
y_train_cat = categorical(y_train_full); % fscmrmr needs categorical Y

final_selected_feature_indices_in_binned_space = [];
final_selected_wavenumbers = [];

if size(X_train_binned, 2) == 0
    error('No features available in training data after binning. Cannot proceed with MRMR.');
end
if final_numMRMRFeatures <= 0
    error('final_numMRMRFeatures must be positive. Value is %d.', final_numMRMRFeatures);
end

try
    [ranked_indices_all_train, ~] = fscmrmr(X_train_binned, y_train_cat);
    
    actual_num_to_select_final = min(final_numMRMRFeatures, length(ranked_indices_all_train));
    if actual_num_to_select_final < final_numMRMRFeatures
        warning('MRMR: Requested %d features, but only %d unique ranked features available/selected from full training set. Using %d.', ...
                final_numMRMRFeatures, length(ranked_indices_all_train), actual_num_to_select_final);
    end
    if actual_num_to_select_final == 0 && final_numMRMRFeatures > 0
        error('MRMR selection resulted in 0 features, though %d were requested. Check data or MRMR scores.', final_numMRMRFeatures);
    elseif actual_num_to_select_final == 0 && final_numMRMRFeatures == 0
        % This case should be caught by the error check above for final_numMRMRFeatures <=0
         error('MRMR: final_numMRMRFeatures is 0.');
    end

    final_selected_feature_indices_in_binned_space = ranked_indices_all_train(1:actual_num_to_select_final);
    final_selected_wavenumbers = wavenumbers_binned(final_selected_feature_indices_in_binned_space);
    
    X_train_fs = X_train_binned(:, final_selected_feature_indices_in_binned_space);
    fprintf('MRMR selected %d features. Final training data for LDA: %d spectra, %d features.\n', ...
        length(final_selected_feature_indices_in_binned_space), size(X_train_fs,1), size(X_train_fs,2));
    
    fprintf('Selected wavenumbers (Top 10 or all if fewer):\n');
    disp(final_selected_wavenumbers(1:min(10, length(final_selected_wavenumbers)))');

catch ME_mrmr_final
    fprintf('ERROR during final MRMR feature selection: %s\n', ME_mrmr_final.message);
    disp(ME_mrmr_final.getReport);
    return;
end

% --- 2.3. Train Final LDA Classifier ---
fprintf('Training final LDA model...\n');
if isempty(X_train_fs) || size(X_train_fs,1) < 2 || length(unique(y_train_full)) < 2
    error('Insufficient data for final LDA training after preprocessing and feature selection.');
end
if size(X_train_fs, 2) == 1 && var(X_train_fs) < 1e-9
    error('Final LDA training: Single feature selected with (near) zero variance.');
end

final_LDA_model = fitcdiscr(X_train_fs, y_train_full);
fprintf('Final LDA model trained.\n');

%% 3. Evaluate Final Model on Test Set
% =========================================================================
fprintf('\n--- Evaluating Final Model on Unseen Test Set ---\n');

% --- 3.1. Apply Binning to Test Set ---
% Use the SAME binningFactor determined from training phase
fprintf('Applying binning (Factor: %d) to test set...\n', final_binningFactor);
if final_binningFactor > 1
    [X_test_binned, ~] = bin_spectra(X_test_full, wavenumbers_original, final_binningFactor);
else
    X_test_binned = X_test_full;
end
fprintf('Test data after binning: %d spectra, %d features.\n', size(X_test_binned,1), size(X_test_binned,2));

% --- 3.2. Apply Feature Selection to Test Set ---
% Use the SAME feature indices selected from the full training set
fprintf('Applying feature selection (using %d features identified from training) to test set...\n', ...
    length(final_selected_feature_indices_in_binned_space));
if isempty(final_selected_feature_indices_in_binned_space) && final_numMRMRFeatures > 0
    error('Final selected feature indices are empty, but features were expected. Cannot process test set.');
elseif isempty(final_selected_feature_indices_in_binned_space) && final_numMRMRFeatures == 0
    X_test_fs = X_test_binned; % No features were meant to be selected
else
    X_test_fs = X_test_binned(:, final_selected_feature_indices_in_binned_space);
end
fprintf('Test data for LDA: %d spectra, %d features.\n', size(X_test_fs,1), size(X_test_fs,2));


% --- 3.3. Predict on Test Set ---
fprintf('Predicting on test set...\n');
if isempty(X_test_fs) && size(X_test_binned,2) > 0 % X_test_fs could be empty if X_test_binned had 0 cols AND selected_indices was 1:0
    warning('Test set has no features after selection. Predictions will likely fail or be trivial.');
    % Depending on LDA behavior with 0 features, might error or give default prediction.
    % For now, let it try to predict.
end

[y_pred_test, y_scores_test] = predict(final_LDA_model, X_test_fs);

% --- 3.4. Calculate Performance Metrics on Test Set ---
fprintf('Calculating performance metrics on test set...\n');
positiveClassLabel_test = 3; % WHO-3
classOrder_test = final_LDA_model.ClassNames;
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

if isempty(positiveClassColIdx_test) || ( ~isempty(positiveClassColIdx_test) && (max(positiveClassColIdx_test) > size(y_scores_test,2)) )
    error('Positive class label %d not found or scores issue in final LDA model for test set evaluation.', positiveClassLabel_test);
end
scores_for_positive_class_test = y_scores_test(:, positiveClassColIdx_test);

testSetPerformanceMetrics = calculate_performance_metrics(y_test_full_numeric, y_pred_test, scores_for_positive_class_test, positiveClassLabel_test, metricNames);

fprintf('\n--- Test Set Performance Metrics ---\n');
metricsTable_test = struct2table(testSetPerformanceMetrics);
disp(metricsTable_test);

%% 4. Save Final Model and Results
% =========================================================================
finalModelPackage = struct();
finalModelPackage.description = 'Final MRMRLDA model trained on the full training set.';
finalModelPackage.trainingDate = string(datetime('now'));
finalModelPackage.LDAModel = final_LDA_model;
finalModelPackage.binningFactor = final_binningFactor;
finalModelPackage.numMRMRFeaturesSelected = length(final_selected_feature_indices_in_binned_space);
finalModelPackage.selectedFeatureIndices_in_binned_space = final_selected_feature_indices_in_binned_space;
finalModelPackage.selectedWavenumbers = final_selected_wavenumbers;
finalModelPackage.originalWavenumbers_before_binning = wavenumbers_original;
finalModelPackage.binnedWavenumbers_for_selection = wavenumbers_binned;
finalModelPackage.testSetPerformance = testSetPerformanceMetrics;
finalModelPackage.trainingDataFile = trainingDataFile;
finalModelPackage.testDataFile = testDataFile;

modelFilename = fullfile(modelsPath, sprintf('%s_Phase3_FinalMRMRLDA_Model.mat', dateStr));
save(modelFilename, 'finalModelPackage');
fprintf('\nFinal model package saved to: %s\n', modelFilename);

resultsFilename_phase3 = fullfile(resultsPath, sprintf('%s_Phase3_TestSetResults.mat', dateStr));
save(resultsFilename_phase3, 'testSetPerformanceMetrics', 'final_binningFactor', 'final_numMRMRFeatures', 'final_selected_wavenumbers');
fprintf('Phase 3 test set results saved to: %s\n', resultsFilename_phase3);

%% 5. Further Analysis / Plotting (Placeholder)
% =========================================================================
% Example: Confusion Matrix for Test Set
if exist('confusionchart','file') % Check if confusionchart is available (newer MATLAB versions)
    figure('Name', 'Confusion Matrix - Test Set');
    cm = confusionchart(y_test_full_numeric, y_pred_test, ...
        'ColumnSummary','column-normalized', ...
        'RowSummary','row-normalized', ...
        'Title', sprintf('Confusion Matrix (Test Set) - MRMRLDA (F2: %.3f)', testSetPerformanceMetrics.F2_WHO3));
    confMatFigFilenameBase = fullfile(figuresPath, sprintf('%s_Phase3_ConfusionMatrix_TestSet', dateStr));
    savefig(gcf, [confMatFigFilenameBase, '.fig']);
    exportgraphics(gcf, [confMatFigFilenameBase, '.tiff'], 'Resolution', 300);
else % Fallback for older MATLAB
    figure('Name', 'Confusion Matrix - Test Set');
    C = confusionmat(y_test_full_numeric, y_pred_test);
    confusionplot(C, final_LDA_model.ClassNames); % confusionplot might be from a toolbox or older
    title(sprintf('Confusion Matrix (Test Set) - MRMRLDA (F2: %.3f)', testSetPerformanceMetrics.F2_WHO3));
    % Add saving logic for this plot if needed
end


fprintf('\nPHASE 3 Processing Complete: %s\n', string(datetime('now')));


%% 5. Probe-Level Aggregation and Visualization for Test Set
% =========================================================================
fprintf('\n--- Aggregating results to Probe-Level for Test Set ---\n');

% Ensure variables from test set prediction are available:
% y_test_full_numeric (true labels for each spectrum)
% probeIDs_test_full (probe ID for each spectrum)
% scores_for_positive_class_test (posterior probability of WHO-3 for each spectrum)
% y_pred_test (predicted class label for each spectrum)

uniqueTestProbes = unique(probeIDs_test_full, 'stable'); % Keep original order from dataTableTest if possible
numUniqueTestProbes = length(uniqueTestProbes);

probeLevelResults = table();
probeLevelResults.Diss_ID = uniqueTestProbes;
probeLevelResults.True_WHO_Grade_Numeric = NaN(numUniqueTestProbes, 1); % Numeric: 1 or 3
probeLevelResults.True_WHO_Grade_Category = categorical(repmat({''}, numUniqueTestProbes, 1));
probeLevelResults.Mean_WHO3_Probability = NaN(numUniqueTestProbes, 1);
probeLevelResults.Predicted_WHO_Grade_Numeric_MeanProb = NaN(numUniqueTestProbes, 1); % Based on mean prob > 0.5
probeLevelResults.MajorityVote_Predicted_WHO_Grade_Numeric = NaN(numUniqueTestProbes, 1);
probeLevelResults.Proportion_Spectra_Predicted_WHO3 = NaN(numUniqueTestProbes, 1);
probeLevelResults.NumSpectraInProbe = NaN(numUniqueTestProbes, 1);

positiveClassLabel = 3; % WHO-3

for i = 1:numUniqueTestProbes
    currentProbeID = uniqueTestProbes{i};
    idxSpectraForProbe = strcmp(probeIDs_test_full, currentProbeID);
    
    probeLevelResults.NumSpectraInProbe(i) = sum(idxSpectraForProbe);
    
    % True WHO Grade for this probe (take from first spectrum, should be consistent)
    true_labels_this_probe = y_test_full_numeric(idxSpectraForProbe);
    if ~isempty(true_labels_this_probe)
        probeLevelResults.True_WHO_Grade_Numeric(i) = mode(true_labels_this_probe); % mode should be fine if consistent
        if mode(true_labels_this_probe) == 1
            probeLevelResults.True_WHO_Grade_Category(i) = 'WHO-1';
        elseif mode(true_labels_this_probe) == 3
            probeLevelResults.True_WHO_Grade_Category(i) = 'WHO-3';
        end
    end
    
    % 1. Mean Probability
    mean_prob_who3 = mean(scores_for_positive_class_test(idxSpectraForProbe));
    probeLevelResults.Mean_WHO3_Probability(i) = mean_prob_who3;
    if mean_prob_who3 > 0.5
        probeLevelResults.Predicted_WHO_Grade_Numeric_MeanProb(i) = positiveClassLabel;
    else
        % Find the alternative class label (assuming binary WHO-1 vs WHO-3)
        otherClasses = setdiff(unique(y_train_full), positiveClassLabel); % Get other class from training labels
        if ~isempty(otherClasses)
            probeLevelResults.Predicted_WHO_Grade_Numeric_MeanProb(i) = otherClasses(1); % Assume 1 for WHO-1
        else
            probeLevelResults.Predicted_WHO_Grade_Numeric_MeanProb(i) = 1; % Default fallback
        end
    end
    
    % 2. Majority Voting
    predicted_labels_this_probe = y_pred_test(idxSpectraForProbe);
    probeLevelResults.MajorityVote_Predicted_WHO_Grade_Numeric(i) = mode(predicted_labels_this_probe);
    probeLevelResults.Proportion_Spectra_Predicted_WHO3(i) = sum(predicted_labels_this_probe == positiveClassLabel) / length(predicted_labels_this_probe);
end

fprintf('\nProbe-Level Aggregated Results (Test Set):\n');
disp(probeLevelResults);

% --- Add results back to dataTableTest (or a copy) ---
% We need to match by Diss_ID. dataTableTest has unique Diss_IDs per row.
dataTableTest_withScores = dataTableTest;
% Pre-allocate new columns
dataTableTest_withScores.Mean_WHO3_Probability = NaN(height(dataTableTest_withScores),1);
dataTableTest_withScores.Predicted_WHO_Grade_MeanProb_Num = NaN(height(dataTableTest_withScores),1);
dataTableTest_withScores.Predicted_WHO_Grade_MajVote_Num = NaN(height(dataTableTest_withScores),1);

for i = 1:height(probeLevelResults)
    probeID_to_match = probeLevelResults.Diss_ID{i};
    idx_in_dataTableTest = strcmp(dataTableTest_withScores.Diss_ID, probeID_to_match);
    if sum(idx_in_dataTableTest) == 1
        dataTableTest_withScores.Mean_WHO3_Probability(idx_in_dataTableTest) = probeLevelResults.Mean_WHO3_Probability(i);
        dataTableTest_withScores.Predicted_WHO_Grade_MeanProb_Num(idx_in_dataTableTest) = probeLevelResults.Predicted_WHO_Grade_Numeric_MeanProb(i);
        dataTableTest_withScores.Predicted_WHO_Grade_MajVote_Num(idx_in_dataTableTest) = probeLevelResults.MajorityVote_Predicted_WHO_Grade_Numeric(i);
    end
end
fprintf('\nScores added to a copy of dataTableTest (dataTableTest_withScores variable).\n');

% --- Plotting Probe-Level Probabilities ---
% (As suggested in your PROMPT.docx: "Probability Distribution Plot")
figure('Name', 'Probe-Level Mean WHO-3 Probabilities (Test Set)', 'Position', [100, 100, 900, 700]);
hold on;

% Define colors (consistent with your prompt)
colorWHO1 = [0.9, 0.6, 0.4]; % Orange
colorWHO3 = [0.4, 0.702, 0.902]; % Blue

% Jitter for better visualization if many points overlap
jitterAmount = 0.02; 

% Separate WHO-1 and WHO-3 true probes
probes_true_who1 = probeLevelResults(probeLevelResults.True_WHO_Grade_Numeric == 1, :);
probes_true_who3 = probeLevelResults(probeLevelResults.True_WHO_Grade_Numeric == 3, :);

h1 = []; h3 = []; % Handles for legend

if ~isempty(probes_true_who1)
    % Create x-coordinates for jitter (randomly around 1)
    x_coords_who1 = 1 + (rand(height(probes_true_who1),1) - 0.5) * jitterAmount * 2;
    h1 = scatter(x_coords_who1, probes_true_who1.Mean_WHO3_Probability, 70, 'o', ...
        'MarkerEdgeColor', 'k', 'MarkerFaceColor', colorWHO1, 'LineWidth', 1, ...
        'DisplayName', 'True WHO-1 Probes');
end

if ~isempty(probes_true_who3)
    % Create x-coordinates for jitter (randomly around 2)
    x_coords_who3 = 2 + (rand(height(probes_true_who3),1) - 0.5) * jitterAmount * 2;
    h3 = scatter(x_coords_who3, probes_true_who3.Mean_WHO3_Probability, 70, 's', ...
        'MarkerEdgeColor', 'k', 'MarkerFaceColor', colorWHO3, 'LineWidth', 1, ...
        'DisplayName', 'True WHO-3 Probes');
end

plot([0.5 2.5], [0.5 0.5], 'k--', 'DisplayName', 'Decision Threshold (0.5)'); % Decision threshold line

% Add text labels for each probe ID
all_probes_sorted = sortrows(probeLevelResults, 'True_WHO_Grade_Numeric');
x_coords_all = [];
y_coords_all = [];
labels_all = {};

if ~isempty(probes_true_who1)
    x_coords_all = [x_coords_all; x_coords_who1];
    y_coords_all = [y_coords_all; probes_true_who1.Mean_WHO3_Probability];
    labels_all = [labels_all; probes_true_who1.Diss_ID];
end
if ~isempty(probes_true_who3)
    x_coords_all = [x_coords_all; x_coords_who3];
    y_coords_all = [y_coords_all; probes_true_who3.Mean_WHO3_Probability];
    labels_all = [labels_all; probes_true_who3.Diss_ID];
end

% Only add text if there are points to label
if ~isempty(x_coords_all)
    text(x_coords_all + 0.02, y_coords_all, labels_all, 'FontSize', 8, 'VerticalAlignment', 'middle');
end

hold off;
xticks([1 2]);
xticklabels({'True WHO-1', 'True WHO-3'});
xlim([0.5 2.5]);
ylim([0 1]);
ylabel('Mean Predicted Probability of being WHO-3');
title({'Probe-Level Classification Results (Test Set)', sprintf('Model DiscrimType: %s, Binning: %d, MRMR Feats: %d', ...
    finalModelPackage.LDAModel.DiscrimType, finalModelPackage.binningFactor, finalModelPackage.numMRMRFeaturesSelected)});
grid on;
legend([h1, h3], 'Location', 'best');
set(gca, 'FontSize', 12);

probeProbFigFilenameBase = fullfile(figuresPath, sprintf('%s_Phase3_ProbeLevelProbabilities_TestSet', dateStr));
savefig(gcf, [probeProbFigFilenameBase, '.fig']);
exportgraphics(gcf, [probeProbFigFilenameBase, '.tiff'], 'Resolution', 300);
fprintf('Probe-level probability plot saved to: %s.(fig/tiff)\n', probeProbFigFilenameBase);


% --- Calculate Probe-Level Performance Metrics ---
% Using Mean Probability with threshold 0.5 for classification
y_true_probe = probeLevelResults.True_WHO_Grade_Numeric;
y_pred_probe_mean_prob = probeLevelResults.Predicted_WHO_Grade_Numeric_MeanProb;
% We need scores for each *probe* for AUC. The Mean_WHO3_Probability is this score.
probe_scores_for_positive_class = probeLevelResults.Mean_WHO3_Probability;

% Remove probes with NaN true labels if any (should not happen if test data is clean)
valid_probes_idx = ~isnan(y_true_probe) & ~isnan(y_pred_probe_mean_prob);
y_true_probe_valid = y_true_probe(valid_probes_idx);
y_pred_probe_mean_prob_valid = y_pred_probe_mean_prob(valid_probes_idx);
probe_scores_for_positive_class_valid = probe_scores_for_positive_class(valid_probes_idx);

if ~isempty(y_true_probe_valid) && length(unique(y_true_probe_valid)) > 1
    fprintf('\n--- Probe-Level Performance Metrics (based on Mean Probability > 0.5) ---\n');
    probeLevelPerfMetrics = calculate_performance_metrics(y_true_probe_valid, y_pred_probe_mean_prob_valid, probe_scores_for_positive_class_valid, positiveClassLabel, metricNames);
    probeMetricsTable = struct2table(probeLevelPerfMetrics);
    disp(probeMetricsTable);

    % Save these probe-level metrics
    probeLevelResultsFilename = fullfile(resultsPath, sprintf('%s_Phase3_ProbeLevelTestSetResults.mat', dateStr));
    save(probeLevelResultsFilename, 'probeLevelResults', 'probeLevelPerfMetrics');
    fprintf('Phase 3 probe-level results and metrics saved to: %s\n', probeLevelResultsFilename);

    % Optional: Confusion Matrix for Probe-Level (Mean Prob)
    if exist('confusionchart','file') && ~isempty(y_true_probe_valid)
        figure('Name', 'Confusion Matrix - Probe-Level (Mean Prob)');
        cm_probe = confusionchart(y_true_probe_valid, y_pred_probe_mean_prob_valid, ...
            'ColumnSummary','column-normalized', ...
            'RowSummary','row-normalized', ...
            'Title', sprintf('Probe-Level Confusion Matrix (Mean Prob > 0.5, F2: %.3f)', probeLevelPerfMetrics.F2_WHO3));
        confMatProbeFigFilenameBase = fullfile(figuresPath, sprintf('%s_Phase3_ConfusionMatrix_ProbeLevel_MeanProb', dateStr));
        savefig(gcf, [confMatProbeFigFilenameBase, '.fig']);
        exportgraphics(gcf, [confMatProbeFigFilenameBase, '.tiff'], 'Resolution', 300);
    end
else
    fprintf('\nCould not calculate probe-level performance metrics (e.g., only one class present or no valid probes).\n');
end

% You could also calculate and display performance for Majority Vote if desired
% y_pred_probe_maj_vote = probeLevelResults.MajorityVote_Predicted_WHO_Grade_Numeric(valid_probes_idx);
% probeLevelPerfMetrics_MajVote = calculate_performance_metrics(y_true_probe_valid, y_pred_probe_maj_vote, [], positiveClassLabel, metricNames); % No scores for maj vote AUC easily
% fprintf('\n--- Probe-Level Performance Metrics (based on Majority Vote) ---\n');
% disp(struct2table(probeLevelPerfMetrics_MajVote));

% ... (previous parts of Section 5 in run_phase3_final_evaluation.m) ...
% Ensure probeLevelResults, scores_for_positive_class_test, probeIDs_test_full,
% finalModelPackage, figuresPath, dateStr are available.

fprintf('\n--- Generating Probe-Level Violin Plot for Test Set ---\n');

% Sort probes for consistent plotting (e.g., by true class, then by ID)
% Group by true class first for visual separation
[~, sort_order_true_class] = sort(probeLevelResults.True_WHO_Grade_Numeric);
sorted_probeLevelResults = probeLevelResults(sort_order_true_class, :);
% Further sort within class by Diss_ID if desired for alphabetical order within groups
% For simplicity, we'll use the order from sorting by True_WHO_Grade_Numeric

numUniqueTestProbes = height(sorted_probeLevelResults);
y_positions = 1:numUniqueTestProbes; % One y-position per probe

% Define colors (consistent with your project)
colorWHO1 = [0.9, 0.6, 0.4]; % Orange for WHO-1
colorWHO3 = [0.4, 0.702, 0.902]; % Blue for WHO-3
colorMisclassified = [0.8, 0, 0]; % Red

figure('Name', 'Probe-Level Spectral Probability Distributions (Test Set)', 'Position', [100, 100, 800, 900]); % Taller figure
hold on;

% Variables to store handles for a custom legend
h_legend = [];
legend_text = {};

% Flags to ensure legend entries are created only once
plotted_legend_who1_true = false;
plotted_legend_who3_true = false;
plotted_legend_misclassified = false;

max_violin_width = 0.4; % Controls the maximum width of the violins

for i = 1:numUniqueTestProbes
    currentProbeID_sorted = sorted_probeLevelResults.Diss_ID{i};
    y_pos = y_positions(i);
    
    % Get data for the current probe from the main sorted table
    probe_data = sorted_probeLevelResults(i,:);
    
    % Get all spectral probabilities for this probe
    idxSpectraForThisProbe = strcmp(probeIDs_test_full, currentProbeID_sorted);
    spectral_probs_this_probe = scores_for_positive_class_test(idxSpectraForThisProbe); % Probabilities of being WHO-3

    % --- 1. True Class Indicator (Left Bar) ---
    true_class_color = colorWHO1;
    if probe_data.True_WHO_Grade_Numeric == 3
        true_class_color = colorWHO3;
    end
    rectangle('Position', [-0.15, y_pos - max_violin_width, 0.05, max_violin_width*2], ...
              'FaceColor', true_class_color, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % --- 2. Predicted Class Indicator (Right Bar, based on Mean_WHO3_Probability) ---
    predicted_class_numeric = probe_data.Predicted_WHO_Grade_Numeric_MeanProb;
    pred_class_color = colorWHO1;
    if predicted_class_numeric == 3
        pred_class_color = colorWHO3;
    end
     rectangle('Position', [1.1, y_pos - max_violin_width, 0.05, max_violin_width*2], ...
              'FaceColor', pred_class_color, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % --- 3. Violin Plot for Spectral Probabilities ---
    if ~isempty(spectral_probs_this_probe)
        % Use ksdensity for the violin shape
        [f_density, xi_density] = ksdensity(spectral_probs_this_probe, 'Support', [0-eps, 1+eps], 'Bandwidth', 0.05); % Small bandwidth for detail
        f_density = f_density / max(f_density) * max_violin_width; % Scale width of violin

        % Plot the violin shape, colored by predicted class of the probe
        fill([xi_density, fliplr(xi_density)], [y_pos + f_density, y_pos - fliplr(f_density)], ...
             pred_class_color, 'FaceAlpha', 0.6, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        
        % Plot median line for the spectral probabilities
        median_prob_spectra = median(spectral_probs_this_probe);
        plot([median_prob_spectra, median_prob_spectra], [y_pos - max_violin_width*0.7, y_pos + max_violin_width*0.7], ...
            'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
    
    % --- 4. Misclassification Asterisk ---
    is_misclassified = (probe_data.True_WHO_Grade_Numeric ~= predicted_class_numeric);
    if is_misclassified
        text(1.06, y_pos, '*', 'Color', colorMisclassified, 'FontSize', 16, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
        if ~plotted_legend_misclassified % For legend
            h_mis_temp = plot(NaN, NaN, '*', 'MarkerSize', 12, 'Color', colorMisclassified, 'LineWidth', 2);
            plotted_legend_misclassified = true;
        end
    end

    % For legend: dummy plots for true class colors
    if probe_data.True_WHO_Grade_Numeric == 1 && ~plotted_legend_who1_true
        h_true1_temp = fill(NaN, NaN, colorWHO1, 'EdgeColor', 'none');
        plotted_legend_who1_true = true;
    elseif probe_data.True_WHO_Grade_Numeric == 3 && ~plotted_legend_who3_true
        h_true3_temp = fill(NaN, NaN, colorWHO3, 'EdgeColor', 'none');
        plotted_legend_who3_true = true;
    end
end

% Add a vertical line for the 0.5 decision threshold for mean probability
h_thresh_line = plot([0.5 0.5], [0 numUniqueTestProbes+1], 'k:'); % Dashed or dotted

hold off;

% --- Axes and Labels ---
set(gca, 'YTick', y_positions);
set(gca, 'YTickLabel', sorted_probeLevelResults.Diss_ID);
set(gca, 'YDir', 'reverse'); 
ylim([0 numUniqueTestProbes+1]);
xlim([-0.2 1.2]); % Adjusted for side bars
xlabel('Probability of WHO-3 Assignment (per spectrum)');
ylabel('Probe ID (Test Set)');
title({'Probe-Level Classification - Spectral Probability Distributions (Test Set)', ...
       sprintf('Final Model: MRMRLDA (Bin=%d, MRMR Feats=%d)', finalModelPackage.binningFactor, finalModelPackage.numMRMRFeaturesSelected)});
grid off; % Grid might be too busy with violins

% --- Construct Legend ---
legend_handles_final = [];
legend_texts_final = {};

if exist('h_true1_temp','var') && isgraphics(h_true1_temp)
    legend_handles_final(end+1) = h_true1_temp;
    legend_texts_final{end+1} = 'True WHO-1';
end
if exist('h_true3_temp','var') && isgraphics(h_true3_temp)
    legend_handles_final(end+1) = h_true3_temp;
    legend_texts_final{end+1} = 'True WHO-3';
end
if exist('h_thresh_line','var') && isgraphics(h_thresh_line)
    legend_handles_final(end+1) = h_thresh_line;
    legend_texts_final{end+1} = 'Decision Threshold (0.5)';
end
if exist('h_mis_temp','var') && isgraphics(h_mis_temp) && plotted_legend_misclassified
    legend_handles_final(end+1) = h_mis_temp;
    legend_texts_final{end+1} = 'Misclassified Probe (by Mean Prob.)';
end

% Add legend entries for violin fill colors (predicted classes)
% Create dummy patches for these as fill objects were 'HandleVisibility','off'
dummy_pred_who1 = fill(NaN,NaN, colorWHO1, 'FaceAlpha', 0.6, 'EdgeColor', 'none');
dummy_pred_who3 = fill(NaN,NaN, colorWHO3, 'FaceAlpha', 0.6, 'EdgeColor', 'none');
legend_handles_final(end+1) = dummy_pred_who1;
legend_texts_final{end+1} = 'Violin/Pred. WHO-1';
legend_handles_final(end+1) = dummy_pred_who3;
legend_texts_final{end+1} = 'Violin/Pred. WHO-3';


if ~isempty(legend_handles_final)
    legend(legend_handles_final, legend_texts_final, 'Location', 'SouthOutside', 'Orientation', 'horizontal', 'NumColumns', 3);
else
    fprintf('Warning: No elements for legend.\n');
end

set(gca, 'FontSize', 10);

% Save the figure
probeViolinFigFilenameBase = fullfile(figuresPath, sprintf('%s_Phase3_ProbeLevelViolinProbabilities_TestSet', dateStr));
savefig(gcf, [probeViolinFigFilenameBase, '.fig']);
exportgraphics(gcf, [probeViolinFigFilenameBase, '.tiff'], 'Resolution', 300);
fprintf('Probe-level violin probability plot saved to: %s.(fig/tiff)\n', probeViolinFigFilenameBase);