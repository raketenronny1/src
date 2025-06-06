function run_phase3_final_evaluation(cfg)
%RUN_PHASE3_FINAL_EVALUATION
%
% Script for Phase 3: Final Model Training & Unbiased Evaluation.
% 1. Trains the best pipeline from Phase 2 (MRMRLDA) on the entire training set.
% 2. Evaluates the final model on the unseen test set.
%
% Date: 2025-05-15

%% 0. Initialization
% =========================================================================
fprintf('PHASE 3: Final Model Training & Unbiased Evaluation - %s\n', string(datetime('now')));

if nargin < 1
    cfg = struct();
end
if ~isfield(cfg, 'projectRoot')
    cfg.projectRoot = pwd;
end

% --- Define Paths (Simplified) ---
P = setup_project_paths(cfg.projectRoot, 'Phase3');
dataPath    = P.dataPath;
resultsPath = P.resultsPath;
modelsPath  = P.modelsPath;
figuresPath = P.figuresPath;

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
metricNames = {'Accuracy', 'Sensitivity_WHO3', 'Specificity_WHO1', 'PPV_WHO3', 'NPV_WHO1', 'F1_WHO3', 'F2_WHO3', 'AUC'};
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

% In run_phase3_final_evaluation.m, replace the existing legend construction for the Violin Plot:

% --- Axes and Labels ---
% (Previous code for setting YTick, YTickLabel, YDir, ylim, xlim, xlabel, ylabel, title, grid for the violin plot axes)
% Let's assume the axes handle for the violin plot is 'ax' (typically gca after plotting the violins)
% If you've explicitly named your violin plot axes, e.g., ax_violin = gca; use ax_violin.
% For this example, I'll assume 'gca' is the correct handle for the violin plot axes at this point.
ax_violin_plot = gca; % Get handle to the current axes where violins were plotted

% --- Construct Legend More Robustly ---
legend_handles_final = gobjects(0); % Initialize as an empty graphics object array
legend_texts_final = {};   % Initialize as an empty cell array

% Conditionally created handles (h_true1_temp, h_true3_temp, h_mis_temp)
% should have been created on ax_violin_plot if they were made within the violin plotting loop.
% h_thresh_line was also plotted on ax_violin_plot.

if exist('h_true1_temp','var') && isgraphics(h_true1_temp)
    legend_handles_final(end+1) = h_true1_temp;
    legend_texts_final{end+1} = 'True WHO-1';
end
if exist('h_true3_temp','var') && isgraphics(h_true3_temp)
    legend_handles_final(end+1) = h_true3_temp;
    legend_texts_final{end+1} = 'True WHO-3';
end

% h_thresh_line should always exist if the plot got this far
if exist('h_thresh_line','var') && isgraphics(h_thresh_line)
    legend_handles_final(end+1) = h_thresh_line;
    legend_texts_final{end+1} = 'Decision Threshold (0.5)';
end

% plotted_legend_misclassified is a flag set if h_mis_temp was created
if exist('plotted_legend_misclassified','var') && plotted_legend_misclassified && exist('h_mis_temp','var') && isgraphics(h_mis_temp)
    legend_handles_final(end+1) = h_mis_temp;
    legend_texts_final{end+1} = 'Misclassified Probe (by Mean Prob.)';
end

% Create dummy patches for violin fill colors, ensuring they are associated with the violin plot's axes
% These are always added.
dummy_pred_who1 = fill(ax_violin_plot, NaN, NaN, colorWHO1, 'FaceAlpha', 0.6, 'EdgeColor', 'none', 'HandleVisibility', 'on'); % Ensure HandleVisibility is on for legend items
dummy_pred_who3 = fill(ax_violin_plot, NaN, NaN, colorWHO3, 'FaceAlpha', 0.6, 'EdgeColor', 'none', 'HandleVisibility', 'on');
legend_handles_final(end+1) = dummy_pred_who1;
legend_texts_final{end+1} = 'Violin/Pred. WHO-1';
legend_handles_final(end+1) = dummy_pred_who3;
legend_texts_final{end+1} = 'Violin/Pred. WHO-3';

% Filter out any invalid handles just before creating the legend
valid_legend_mask = isgraphics(legend_handles_final);
if any(~valid_legend_mask)
    fprintf('Warning: Some handles collected for the violin plot legend were invalid. Filtering them out.\n');
    disp('Indices of invalid handles before filtering:');
    disp(find(~valid_legend_mask));
    legend_handles_final = legend_handles_final(valid_legend_mask);
    legend_texts_final = legend_texts_final(valid_legend_mask);
end

% Create the legend if there are valid items
if ~isempty(legend_handles_final) && ~isempty(legend_texts_final) && (length(legend_handles_final) == length(legend_texts_final))
    try
        legend(ax_violin_plot, legend_handles_final, legend_texts_final, 'Location', 'SouthOutside', 'Orientation', 'horizontal', 'NumColumns', 3);
        fprintf('Violin plot legend created successfully.\n');
    catch ME_legend
        fprintf('ERROR creating violin plot legend: %s\n', ME_legend.message);
        disp('Final legend_handles_final:'); disp(legend_handles_final);
        disp('Final legend_texts_final:'); disp(legend_texts_final);
    end
else
    fprintf('Warning: No valid elements for violin plot legend or mismatch between handles and texts. Legend skipped.\n');
    if isempty(legend_handles_final)
        disp('Reason: legend_handles_final is empty.');
    elseif isempty(legend_texts_final)
        disp('Reason: legend_texts_final is empty.');
    else
        fprintf('Reason: Mismatch in lengths. Handles: %d, Texts: %d.\n', length(legend_handles_final), length(legend_texts_final));
    end
end

set(gca, 'FontSize', 10); % This was 'set(gca, 'FontSize', 10);' in your original code, ensure gca is ax_violin_plot here

% Save the figure (ensure figViolin is the correct handle for the figure containing the violin plot)
% Assuming the figure was created with: figViolin = figure(...);
probeViolinFigFilenameBase = fullfile(figuresPath, sprintf('%s_Phase3_ProbeLevelViolinProbabilities_TestSet', dateStr));
% Ensure 'figViolin' is the correct figure handle for the violin plot
current_fig_handle_for_violin = gcf; % Or if you stored it e.g., fig_violin_plot_handle = figure(...);
if isgraphics(current_fig_handle_for_violin)
    try
        savefig(current_fig_handle_for_violin, [probeViolinFigFilenameBase, '.fig']);
        exportgraphics(current_fig_handle_for_violin, [probeViolinFigFilenameBase, '.tiff'], 'Resolution', 300);
        fprintf('Probe-level violin probability plot saved to: %s.(fig/tiff)\n', probeViolinFigFilenameBase);
    catch ME_save_violin
         fprintf('ERROR saving violin plot: %s\n', ME_save_violin.message);
    end
else
    fprintf('WARNING: Violin plot figure handle is invalid. Plot not saved.\n');
end

%% 6. Additional Visualizations (Spectrum-Level ROC and DET Curves)
% =========================================================================
fprintf('\n--- Generating Spectrum-Level ROC and DET Curves ---\n');

if exist('y_test_full_numeric', 'var') && exist('scores_for_positive_class_test', 'var') && ...
   ~isempty(y_test_full_numeric) && ~isempty(scores_for_positive_class_test) && exist('final_LDA_model','var')

    positiveClassLabel_for_curves = 3; % Assuming WHO-3 is numerically 3. Adjust if different.
    % Ensure this matches the positiveClassLabel_test used for metrics calculation earlier.

    % Define class names if you want them in titles, though not directly used by plotRocAndDetCurves
    % classNames_for_plot = final_LDA_model.ClassNames; % Or define manually: {'WHO I', 'WHO III'}
    % If classNames_for_plot is categorical, convert positiveClassLabel_for_curves if needed for titles.
    
    titleRocStr = sprintf('ROC (Test Set, Spectrum-Level, Bin:%d, MRMR:%d)', ...
                          finalModelPackage.binningFactor, finalModelPackage.numMRMRFeaturesSelected);
    titleDetStr = sprintf('DET (Test Set, Spectrum-Level, Bin:%d, MRMR:%d)', ...
                          finalModelPackage.binningFactor, finalModelPackage.numMRMRFeaturesSelected);

    % Call the plotting function
    % The last argument 'true' enables the normal deviate scale for the DET curve.
    fh_roc_det_spectrum = plotRocAndDetCurves(y_test_full_numeric, ...
                                              scores_for_positive_class_test, ...
                                              positiveClassLabel_for_curves, ...
                                              titleRocStr, ...
                                              titleDetStr, ...
                                              true); % Use normal deviate scale for DET

    if isgraphics(fh_roc_det_spectrum) % Check if figure was created
        rocDetFigFilenameBase = fullfile(figuresPath, sprintf('%s_Phase3_ROC_DET_SpectrumLevel', dateStr));
        try
            exportgraphics(fh_roc_det_spectrum, [rocDetFigFilenameBase, '.tiff'], 'Resolution', 300);
            savefig(fh_roc_det_spectrum, [rocDetFigFilenameBase, '.fig']);
            fprintf('Spectrum-Level ROC and DET curves saved to: %s.(fig/tiff)\n', rocDetFigFilenameBase);
        catch ME_save_roc
            fprintf('Warning: Could not save ROC/DET plot: %s\n', ME_save_roc.message);
        end
        % close(fh_roc_det_spectrum); % Optional: close after saving
    else
        fprintf('Spectrum-Level ROC and DET curves were not generated.\n');
    end
else
    fprintf('Skipping Spectrum-Level ROC/DET curves: Required data (true labels, scores, or model) not available.\n');
    if ~exist('y_test_full_numeric', 'var') || isempty(y_test_full_numeric)
        disp('Reason: y_test_full_numeric not found or empty.');
    end
    if ~exist('scores_for_positive_class_test', 'var') || isempty(scores_for_positive_class_test)
        disp('Reason: scores_for_positive_class_test not found or empty.');
    end
     if ~exist('final_LDA_model', 'var')
        disp('Reason: final_LDA_model not found (needed for context in titles via finalModelPackage).');
    end
end


fprintf('\nPHASE 3 Processing Complete (including additional visualizations): %s\n', string(datetime('now')));


% =========================================================================
% HELPER FUNCTION DEFINITIONS
% =========================================================================
% Make sure 'calculate_performance_metrics.m' and 'bin_spectra.m' are in your 'helper_functions' path.
% Add the plotRocAndDetCurves function definition here if it's not already in 'helper_functions'
% or at the end of your main script.

% --- START: Definition of plotRocAndDetCurves (if not in helper_functions) ---
% (Paste the function definition provided in the previous response here)
function fh_roc_det = plotRocAndDetCurves(trueLabels, scores, positiveClassIdentifier, titleRocStr, titleDetStr, useNormalDeviateScaleDet)
% plotRocAndDetCurves - Plots ROC and DET curves side-by-side.
% Inputs:
%   trueLabels - Vector of true binary labels (e.g., 0/1, logical, or categorical with two classes).
%                If categorical, positiveClassIdentifier must match one of the categories.
%                If numeric/logical, positiveClassIdentifier must match the value of the positive class.
%   scores - Vector of scores or probabilities for the positive class.
%   positiveClassIdentifier - The label of the positive class (e.g., 1, true, 'WHO III').
%   titleRocStr - String for the ROC plot title.
%   titleDetStr - String for the DET plot title.
%   useNormalDeviateScaleDet - Logical, true to use normal deviate scale for DET axes.

    if nargin < 4 || isempty(titleRocStr)
        titleRocStr = 'ROC Curve';
    end
    if nargin < 5 || isempty(titleDetStr)
        titleDetStr = 'DET Curve';
    end
    if nargin < 6 || isempty(useNormalDeviateScaleDet)
        useNormalDeviateScaleDet = false;
    end

    fh_roc_det = figure('Name', [titleRocStr, ' & ', titleDetStr], 'NumberTitle', 'off', 'Visible', 'on'); % Ensure visible
    try
        % perfcurve can handle numeric, logical, or categorical trueLabels.
        % If trueLabels are categorical, positiveClassIdentifier should be one of the category names (string or char).
        % If trueLabels are numeric/logical, positiveClassIdentifier should be the numeric/logical value of the positive class.
        [Xroc, Yroc, ~, AUCroc] = perfcurve(trueLabels, scores, positiveClassIdentifier);
        [Xdet_fpr, Ydet_fnr] = perfcurve(trueLabels, scores, positiveClassIdentifier, 'XCrit', 'fpr', 'YCrit', 'fnr');

        tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        ax_roc = nexttile;
        plot(ax_roc, Xroc, Yroc, 'LineWidth', 1.5);
        hold(ax_roc, 'on'); plot(ax_roc, [0 1], [0 1], 'k--'); hold(ax_roc, 'off');
        xlabel(ax_roc, 'False Positive Rate (FPR)'); ylabel(ax_roc, 'True Positive Rate (TPR)');
        title(ax_roc, sprintf('%s (AUC = %.3f)', titleRocStr, AUCroc));
        grid(ax_roc, 'on'); axis(ax_roc, [0 1 0 1]);

        ax_det = nexttile;
        if useNormalDeviateScaleDet
            epsilon = 1e-7; 
            Xdet_fpr_safe = max(epsilon, min(1-epsilon, Xdet_fpr));
            Ydet_fnr_safe = max(epsilon, min(1-epsilon, Ydet_fnr));
            Xdet_fpr_norm = norminv(Xdet_fpr_safe);
            Ydet_fnr_norm = norminv(Ydet_fnr_safe);
            
            plot(ax_det, Xdet_fpr_norm, Ydet_fnr_norm, 'LineWidth', 1.5);
            xlabel(ax_det, 'FPR (Normal Deviate Scale)'); ylabel(ax_det, 'FNR (Normal Deviate Scale)');
            title(ax_det, [titleDetStr, ' (Normal Deviate)']);
            
            prob_ticks = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999];
            norm_ticks = norminv(prob_ticks);
            x_ticks_to_use = norm_ticks(norm_ticks >= min(Xdet_fpr_norm(isfinite(Xdet_fpr_norm))) & norm_ticks <= max(Xdet_fpr_norm(isfinite(Xdet_fpr_norm))));
            y_ticks_to_use = norm_ticks(norm_ticks >= min(Ydet_fnr_norm(isfinite(Ydet_fnr_norm))) & norm_ticks <= max(Ydet_fnr_norm(isfinite(Ydet_fnr_norm))));

            if ~isempty(x_ticks_to_use), set(ax_det, 'XTick', x_ticks_to_use, 'XTickLabel', sprintfc('%.3g', prob_ticks(ismembertol(norm_ticks, x_ticks_to_use)))); end
            if ~isempty(y_ticks_to_use), set(ax_det, 'YTick', y_ticks_to_use, 'YTickLabel', sprintfc('%.3g', prob_ticks(ismembertol(norm_ticks, y_ticks_to_use)))); end
        else
            plot(ax_det, Xdet_fpr, Ydet_fnr, 'LineWidth', 1.5);
            xlabel(ax_det, 'False Positive Rate (FPR)'); ylabel(ax_det, 'False Negative Rate (FNR)');
            title(ax_det, titleDetStr); axis(ax_det, [0 1 0 1]);
        end
        grid(ax_det, 'on');
        fprintf('ROC and DET curves ''%s'' & ''%s'' plotted.\n', titleRocStr, titleDetStr);
    catch ME
        fprintf('Error plotting ROC/DET curves: %s\n', ME.message);
        disp(ME.getReport); % Display more detailed error information
        if ishandle(fh_roc_det); close(fh_roc_det); fh_roc_det = []; end
    end
end
% --- END: Definition of plotRocAndDetCurves ---
end
