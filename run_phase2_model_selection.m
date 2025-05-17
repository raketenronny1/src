% run_phase2_model_selection.m
%
% Main script for Phase 2: Model and Feature Selection Pipelines.
% Implements nested cross-validation to find the best combination of
% preprocessing (binning), feature selection, and classifier for
% WHO-1 vs WHO-3 meningioma classification.
%
% Focus: Prioritize correct identification of WHO-3.
% Method: Probe-wise 5-fold nested cross-validation.
%
% Date: 2025-05-15 (Updated with fscmrmr fix)

%% 0. Initialization
% ... (your existing setup) ...
fprintf('PHASE 2: Model and Feature Selection - %s\n', string(datetime('now')));

% +++ NEW: CHOOSE OUTLIER STRATEGY FOR THIS RUN +++
% outlierStrategy = 'OR'; % Options: 'OR' or 'AND'
outlierStrategy = 'AND'; % Or set this from a config file P.outlier_cleaning_for_phase2

fprintf('--- Using outlier removal strategy: T2 %s Q ---\n', outlierStrategy);
% ++++++++++++++++++++++++++++++++++++++++++++++++++

% ... (paths setup) ...
dateStr = string(datetime('now','Format','yyyyMMdd')); % Keep local dateStr for output naming

%% 1. Load Data
% ...
if strcmpi(outlierStrategy, 'OR')
    inputDataFilePattern = '*_training_set_no_outliers_T2orQ.mat';
elseif strcmpi(outlierStrategy, 'AND')
    inputDataFilePattern = '*_training_set_no_outliers_T2andQ.mat';
else
    error('Invalid outlierStrategy specified. Choose "OR" or "AND".');
end

cleanedDataFiles = dir(fullfile(dataPath, inputDataFilePattern));
if isempty(cleanedDataFiles)
    error('No cleaned training set file found for strategy "%s" in %s matching pattern %s', ...
          outlierStrategy, dataPath, inputDataFilePattern);
end
[~,idxSortCleaned] = sort([cleanedDataFiles.datenum],'descend');
inputDataFile = fullfile(dataPath, cleanedDataFiles(idxSortCleaned(1)).name);
fprintf('Loading cleaned training data (Strategy: %s) from: %s\n', outlierStrategy, inputDataFile);

try
    if strcmpi(outlierStrategy, 'OR')
        loadedData = load(inputDataFile, ...
                           'X_train_no_outliers_OR', 'y_train_numeric_no_outliers_OR', ...
                           'Patient_ID_train_no_outliers_OR', 'wavenumbers_roi');
        X_train_full = loadedData.X_train_no_outliers_OR;
        y_train_full = loadedData.y_train_numeric_no_outliers_OR; % Ensure this is numeric for CV
        probeIDs_train_full = loadedData.Patient_ID_train_no_outliers_OR;
    else % AND strategy
        loadedData = load(inputDataFile, ...
                           'X_train_no_outliers_AND', 'y_train_numeric_no_outliers_AND', ...
                           'Patient_ID_train_no_outliers_AND', 'wavenumbers_roi');
        X_train_full = loadedData.X_train_no_outliers_AND;
        y_train_full = loadedData.y_train_numeric_no_outliers_AND; % Ensure this is numeric
        probeIDs_train_full = loadedData.Patient_ID_train_no_outliers_AND;
    end
    
    wavenumbers_original = loadedData.wavenumbers_roi; % Common
    % ... rest of your data loading/verification ...
catch ME
    fprintf('ERROR loading data from %s: %s\n', inputDataFile, ME.message);
    rethrow(ME);
end

%% 2. Define Cross-Validation Parameters
% =========================================================================
numOuterFolds = 5;
rng('default');

[uniqueProbes, ~, groupIdxPerSpectrum] = unique(probeIDs_train_full, 'stable');
probe_WHO_Grade = zeros(length(uniqueProbes), 1);
for i = 1:length(uniqueProbes)
    probeSpectraLabels = y_train_full(groupIdxPerSpectrum == i);
    if any(probeSpectraLabels == 3)
        probe_WHO_Grade(i) = 3;
    else
        probe_WHO_Grade(i) = mode(probeSpectraLabels);
    end
end
outerCV_probeLevel = cvpartition(probe_WHO_Grade, 'KFold', numOuterFolds);

numInnerFolds = 3;
metricNames = {'Accuracy', 'Sensitivity_WHO3', 'Specificity_WHO1', 'PPV_WHO3', 'NPV_WHO1', 'F1_WHO3', 'F2_WHO3', 'AUC'};

%% 3. Define Pipelines to Evaluate
% =========================================================================
pipelines = cell(0,1); 
pipelineIdx = 0;

% --- Pipeline 1: Baseline (Binning) + LDA ---
p = struct();
p.name = 'BaselineLDA';
p.feature_selection_method = 'none';
p.classifier = 'LDA';
p.hyperparameters_to_tune = {'binningFactor'};
p.binningFactors = [1, 2, 4, 8, 16];
pipelineIdx = pipelineIdx + 1;
pipelines{pipelineIdx} = p;

% --- Pipeline 2: Fisher Ratio + LDA ---
p = struct();
p.name = 'FisherLDA';
p.feature_selection_method = 'fisher';
p.classifier = 'LDA';
p.hyperparameters_to_tune = {'binningFactor', 'numFisherFeatures'};
p.binningFactors = [1, 2, 4, 8, 16];
p.numFisherFeatures_range = [10, 20, 30, 40, 50, 75, 100];
pipelineIdx = pipelineIdx + 1;
pipelines{pipelineIdx} = p;

% --- Pipeline 3: PCA + LDA ---
p = struct();
p.name = 'PCALDA';
p.feature_selection_method = 'pca';
p.classifier = 'LDA';
p.hyperparameters_to_tune = {'binningFactor', 'pcaVarianceToExplain'};
p.binningFactors = [1, 2, 4, 8, 16];
p.pcaVarianceToExplain_range = [0.90, 0.95, 0.99];
pipelineIdx = pipelineIdx + 1;
pipelines{pipelineIdx} = p;

% --- Pipeline 4: MRMR + LDA ---
p = struct();
p.name = 'MRMRLDA';
p.feature_selection_method = 'mrmr';
p.classifier = 'LDA';
p.hyperparameters_to_tune = {'binningFactor', 'numMRMRFeatures'};
p.binningFactors = [1, 2, 4, 8, 16];
p.numMRMRFeatures_range = [10, 20, 30, 40, 50];
pipelineIdx = pipelineIdx + 1;
pipelines{pipelineIdx} = p;

allPipelinesResults = cell(length(pipelines), 1);

%% 4. Nested Cross-Validation Loop
% =========================================================================
fprintf('\nStarting Nested Cross-Validation...\n');

for iPipeline = 1:length(pipelines)
    currentPipeline = pipelines{iPipeline}; % CORRECTED: Use {} for cell access
    fprintf('\n--- Evaluating Pipeline: %s ---\n', currentPipeline.name);

    outerFoldMetrics = NaN(numOuterFolds, length(metricNames));
    outerFoldBestHyperparams = cell(numOuterFolds, 1);
    outerFoldModels = cell(numOuterFolds, 1);
    outerFoldSelectedFeaturesInfo = cell(numOuterFolds, 1);

    for kOuter = 1:numOuterFolds
        fprintf('  Outer Fold %d/%d:\n', kOuter, numOuterFolds);

        isOuterTrainProbe_IndicesInUniqueList = training(outerCV_probeLevel, kOuter);
        isOuterTestProbe_IndicesInUniqueList  = test(outerCV_probeLevel, kOuter);

        outerTrainProbeIDs = uniqueProbes(isOuterTrainProbe_IndicesInUniqueList);
        outerTestProbeIDs  = uniqueProbes(isOuterTestProbe_IndicesInUniqueList);

        idxOuterTrain = ismember(probeIDs_train_full, outerTrainProbeIDs);
        idxOuterTest  = ismember(probeIDs_train_full, outerTestProbeIDs);

        X_outer_train = X_train_full(idxOuterTrain, :);
        y_outer_train = y_train_full(idxOuterTrain);
        probeIDs_outer_train = probeIDs_train_full(idxOuterTrain);

        X_outer_test  = X_train_full(idxOuterTest, :);
        y_outer_test  = y_train_full(idxOuterTest);

        fprintf('    Outer train: %d spectra from %d probes. Outer test: %d spectra from %d probes.\n', ...
            size(X_outer_train,1), length(outerTrainProbeIDs), ...
            size(X_outer_test,1), length(outerTestProbeIDs));
        
        if isempty(X_outer_train) || isempty(X_outer_test)
            fprintf('    WARNING: Outer fold %d has empty training or test set after probe split. Skipping fold.\n', kOuter);
            outerFoldMetrics(kOuter, :) = NaN;
            continue;
        end
        if length(unique(y_outer_train)) < 2
             fprintf('    WARNING: Outer fold %d training data has only one class. Skipping fold.\n', kOuter);
             outerFoldMetrics(kOuter, :) = NaN;
             continue;
        end

        [bestHyperparams, bestInnerPerf] = perform_inner_cv(...
            X_outer_train, y_outer_train, probeIDs_outer_train, ...
            currentPipeline, wavenumbers_original, numInnerFolds, metricNames);

        fprintf('    Best hyperparameters from inner CV: '); disp(bestHyperparams);
        if isstruct(bestInnerPerf) && isfield(bestInnerPerf, 'F2_WHO3')
             fprintf('    Corresponding inner CV performance (F2_WHO3): %.4f\n', bestInnerPerf.F2_WHO3);
        else
             fprintf('    Inner CV performance could not be determined or F2_WHO3 not available.\n');
        end
        outerFoldBestHyperparams{kOuter} = bestHyperparams;

        currentWavenumbers_fold = wavenumbers_original;
        X_train_processed = X_outer_train;
        X_test_processed = X_outer_test;  

        if isfield(bestHyperparams, 'binningFactor') && bestHyperparams.binningFactor > 1
            [X_train_processed, currentWavenumbers_fold] = bin_spectra(X_outer_train, wavenumbers_original, bestHyperparams.binningFactor);
            [X_test_processed, ~] = bin_spectra(X_outer_test, wavenumbers_original, bestHyperparams.binningFactor);
        end

        selectedFeatureIndices_in_current_w = 1:size(X_train_processed, 2);
        feature_selection_model_info = struct('method', currentPipeline.feature_selection_method);

        switch lower(currentPipeline.feature_selection_method) % Use lower for case-insensitivity
            case 'fisher'
                fisherRatios = calculate_fisher_ratio(X_train_processed, y_outer_train);
                [~, sorted_indices] = sort(fisherRatios, 'descend', 'MissingPlacement','last');
                numFeat = min(bestHyperparams.numFisherFeatures, length(sorted_indices));
                selectedFeatureIndices_in_current_w = sorted_indices(1:numFeat);
                feature_selection_model_info.num_selected = numFeat;
                fprintf('    Fisher: Selected %d features for outer fold.\n', numFeat);
            case 'pca'
                [coeff_pca, score_train_pca, ~, ~, explained_pca, mu_pca] = pca(X_train_processed);
                cumulativeExplained = cumsum(explained_pca);
                if isfield(bestHyperparams, 'pcaVarianceToExplain')
                    numComponents = find(cumulativeExplained >= bestHyperparams.pcaVarianceToExplain*100, 1, 'first');
                     if isempty(numComponents), numComponents = size(coeff_pca,2); end
                else 
                    numComponents = min(bestHyperparams.numPCAComponents, size(coeff_pca,2));
                end
                
                if numComponents == 0
                    fprintf('    PCA: Resulted in 0 components. Using all %d original (binned) features instead for this fold.\n', size(X_train_processed,2));
                    feature_selection_model_info.method = 'pca_fallback_none';
                else
                    X_train_processed = score_train_pca(:, 1:numComponents);
                    X_test_processed = (X_test_processed - mu_pca) * coeff_pca(:, 1:numComponents);
                    selectedFeatureIndices_in_current_w = 1:numComponents;
                    feature_selection_model_info.coeff = coeff_pca(:, 1:numComponents);
                    feature_selection_model_info.mu = mu_pca;
                    feature_selection_model_info.num_components = numComponents;
                    if numComponents <= length(cumulativeExplained) % Ensure index is valid
                        feature_selection_model_info.explained_variance_by_selected_components = cumulativeExplained(numComponents);
                        fprintf('    PCA: Selected %d components (explaining %.2f%% variance) for outer fold.\n', numComponents, cumulativeExplained(numComponents)); % CORRECTED PRINTOUT
                    else % Should not happen if numComponents derived from explained_pca
                         fprintf('    PCA: Selected %d components, but variance calculation index out of bounds.\n', numComponents);
                         feature_selection_model_info.explained_variance_by_selected_components = NaN;
                    end
                end
            case 'mrmr'
                y_outer_train_cat = categorical(y_outer_train);
                numFeatToSelect = min(bestHyperparams.numMRMRFeatures, size(X_train_processed,2));

                fprintf('DEBUG MRMR (run_phase2): size(X_train_processed) = [%d, %d], class = %s\n', size(X_train_processed,1), size(X_train_processed,2), class(X_train_processed));
                fprintf('DEBUG MRMR (run_phase2): size(y_outer_train_cat) = [%d, %d], class = %s\n', size(y_outer_train_cat,1), size(y_outer_train_cat,2), class(y_outer_train_cat));
                fprintf('DEBUG MRMR (run_phase2): numFeatToSelect (target) = %f, class = %s\n', numFeatToSelect, class(numFeatToSelect));
                
                if size(X_train_processed,2) < numFeatToSelect && size(X_train_processed,2) > 0
                    warning('DEBUG MRMR (run_phase2): Number of features in X (%d) is less than numFeatToSelect (%d)! Adjusting.', size(X_train_processed,2), numFeatToSelect);
                    numFeatToSelect = size(X_train_processed,2);
                    fprintf('DEBUG MRMR (run_phase2): Adjusted numFeatToSelect = %f\n', numFeatToSelect);
                end

                if numFeatToSelect <=0 || size(X_train_processed,2) == 0
                     fprintf('DEBUG MRMR (run_phase2): numFeatToSelect is %d or no features in X. Using all available features.\n', numFeatToSelect);
                     selectedFeatureIndices_in_current_w = 1:size(X_train_processed,2);
                     feature_selection_model_info.method = 'mrmr_fallback_no_features_to_select_or_no_X_features';
                     if size(X_train_processed,2) > 0
                         feature_selection_model_info.num_selected = length(selectedFeatureIndices_in_current_w);
                     else
                         feature_selection_model_info.num_selected = 0;
                     end
                else
                    try
                        [ranked_indices, ~] = fscmrmr(X_train_processed, y_outer_train_cat); % Get all features ranked
                        actual_num_to_take = min(numFeatToSelect, length(ranked_indices)); 
                        if actual_num_to_take > 0
                            selectedFeatureIndices_in_current_w = ranked_indices(1:actual_num_to_take);
                        else 
                            selectedFeatureIndices_in_current_w = 1:size(X_train_processed,2); 
                            fprintf('    MRMR: actual_num_to_take is %d. Fallback to all %d features.\n', actual_num_to_take, size(X_train_processed,2));
                        end
                        feature_selection_model_info.num_selected = length(selectedFeatureIndices_in_current_w);
                        fprintf('    MRMR: Selected %d features for outer fold.\n', length(selectedFeatureIndices_in_current_w));
                    catch ME_fscmrmr_main
                        fprintf('ERROR during fscmrmr(X,Y) in run_phase2: %s. Using all features.\n', ME_fscmrmr_main.message);
                        selectedFeatureIndices_in_current_w = 1:size(X_train_processed,2); 
                        feature_selection_model_info.num_selected = length(selectedFeatureIndices_in_current_w);
                        feature_selection_model_info.method = 'mrmr_fallback_error_in_fscmrmrXY';
                    end
                end
            case 'none'
                fprintf('    No explicit feature selection beyond binning for outer fold.\n');
        end
        
        if isempty(selectedFeatureIndices_in_current_w) && ~(contains(feature_selection_model_info.method, 'fallback') || size(X_train_processed,2)==0)
            fprintf('    WARNING: Feature selection resulted in zero features. Using all %d (binned) features for this fold.\n', size(X_train_processed,2));
            selectedFeatureIndices_in_current_w = 1:size(X_train_processed, 2);
            feature_selection_model_info.method = [currentPipeline.feature_selection_method, '_fallback_all'];
        end

        X_fs_train = X_train_processed(:, selectedFeatureIndices_in_current_w);
        X_fs_test  = X_test_processed(:, selectedFeatureIndices_in_current_w);
        
        feature_selection_model_info.selected_indices_in_binned_space = selectedFeatureIndices_in_current_w;
        if ~strcmpi(currentPipeline.feature_selection_method, 'pca') && ~isempty(selectedFeatureIndices_in_current_w) && max(selectedFeatureIndices_in_current_w) <= length(currentWavenumbers_fold)
            feature_selection_model_info.selected_wavenumbers = currentWavenumbers_fold(selectedFeatureIndices_in_current_w);
        elseif strcmpi(currentPipeline.feature_selection_method, 'pca')
            feature_selection_model_info.original_wavenumbers_for_pca_input = currentWavenumbers_fold;
        else
             feature_selection_model_info.selected_wavenumbers = []; % Not applicable or error
        end
        outerFoldSelectedFeaturesInfo{kOuter} = feature_selection_model_info;

        trainedClassifier = [];
        if isempty(X_fs_train) || size(X_fs_train,1) < 2 || length(unique(y_outer_train)) < 2
            fprintf('    Skipping classifier training for outer fold %d: insufficient data after preproc/FS.\n', kOuter);
            outerFoldMetrics(kOuter, :) = NaN;
            continue;
        end

        switch lower(currentPipeline.classifier) % Use lower for case-insensitivity
            case 'lda'
                if size(X_fs_train, 2) == 1 && var(X_fs_train) < 1e-9
                    warning('LDA training: Single feature with (near) zero variance in outer fold %d. Skipping LDA for this fold.', kOuter);
                    outerFoldMetrics(kOuter, :) = NaN;
                    continue;
                end
                trainedClassifier = fitcdiscr(X_fs_train, y_outer_train);
        end
        outerFoldModels{kOuter} = trainedClassifier;

        if ~isempty(trainedClassifier) && ~isempty(X_fs_test) && ~isempty(y_outer_test)
            [y_pred_outer, y_scores_outer] = predict(trainedClassifier, X_fs_test);
            positiveClassLabel = 3;
            classOrder = trainedClassifier.ClassNames;
            positiveClassColIdx = [];
            if isnumeric(classOrder) && isnumeric(positiveClassLabel)
                 positiveClassColIdx = find(classOrder == positiveClassLabel);
            elseif iscategorical(classOrder) && isnumeric(positiveClassLabel)
                 positiveClassColIdx = find(str2double(string(classOrder)) == positiveClassLabel);
            elseif iscellstr(classOrder) && isnumeric(positiveClassLabel) % Check if classOrder is cell array of strings
                 positiveClassColIdx = find(str2double(classOrder) == positiveClassLabel);
            else 
                 positiveClassColIdx = find(classOrder == positiveClassLabel);
            end

            if isempty(positiveClassColIdx) || size(y_scores_outer,2) < max(positiveClassColIdx)
                 warning('Positive class label %d not found or scores issue in trained classifier for outer fold %d.', positiveClassLabel, kOuter);
                 outerFoldMetrics(kOuter, :) = NaN;
                 continue;
            end
            scores_for_positive_class = y_scores_outer(:, positiveClassColIdx);
            currentFoldMetricsStruct = calculate_performance_metrics(y_outer_test, y_pred_outer, scores_for_positive_class, positiveClassLabel, metricNames);
            outerFoldMetrics(kOuter, :) = cell2mat(struct2cell(currentFoldMetricsStruct))';
            fprintf('    Outer fold %d test metrics: Acc=%.3f, Sens_WHO3=%.3f, Spec_WHO1=%.3f, F2_WHO3=%.3f\n', ...
                kOuter, currentFoldMetricsStruct.Accuracy, currentFoldMetricsStruct.Sensitivity_WHO3, currentFoldMetricsStruct.Specificity_WHO1, currentFoldMetricsStruct.F2_WHO3);
        else
             outerFoldMetrics(kOuter, :) = NaN;
             if isempty(trainedClassifier)
                 fprintf('    Outer fold %d: Evaluation skipped (classifier not trained).\n', kOuter);
             elseif isempty(X_fs_test)
                 fprintf('    Outer fold %d: Evaluation skipped (test set empty after processing).\n', kOuter);
             end
        end
    end 

    meanOuterFoldMetrics = nanmean(outerFoldMetrics, 1);
    stdOuterFoldMetrics = nanstd(outerFoldMetrics, 0, 1);

    fprintf('  --- Pipeline %s Average Performance (over %d outer folds) ---\n', currentPipeline.name, numOuterFolds);
    metricsTable = table(metricNames(:), meanOuterFoldMetrics(:), stdOuterFoldMetrics(:), ... % Ensure column vectors
        'VariableNames', {'Metric', 'Mean', 'StdDev'});
    disp(metricsTable);

    pipelineSummary = struct();
    pipelineSummary.pipelineConfig = currentPipeline;
    pipelineSummary.outerFoldMetrics_raw = outerFoldMetrics;
    pipelineSummary.outerFoldMetrics_mean = meanOuterFoldMetrics;
    pipelineSummary.outerFoldMetrics_std = stdOuterFoldMetrics;
    pipelineSummary.outerFoldBestHyperparams = outerFoldBestHyperparams;
    pipelineSummary.outerFoldSelectedFeaturesInfo = outerFoldSelectedFeaturesInfo;
    pipelineSummary.metricNames = metricNames;
    allPipelinesResults{iPipeline} = pipelineSummary;
end

%% 5. Select Best Overall Pipeline and Save Results
% =========================================================================
fprintf('\n--- Selecting Best Overall Pipeline (Outlier Strategy: %s) ---\n', outlierStrategy); % Added strategy to log
bestF2Score = -Inf;
bestPipelineIdx = -1;
f2_idx = find(strcmpi(metricNames, 'F2_WHO3')); % Use strcmpi for case-insensitivity
if isempty(f2_idx)
    error('F2_WHO3 metric not found in metricNames. Check definition.');
end

for iPipeline = 1:length(pipelines)
    if ~isempty(allPipelinesResults{iPipeline}) && isstruct(allPipelinesResults{iPipeline}) && isfield(allPipelinesResults{iPipeline}, 'outerFoldMetrics_mean') ...
            && length(allPipelinesResults{iPipeline}.outerFoldMetrics_mean) >= f2_idx
        currentMeanF2 = allPipelinesResults{iPipeline}.outerFoldMetrics_mean(f2_idx);
        fprintf('Pipeline: %s, Mean F2_WHO3: %.4f\n', pipelines{iPipeline}.name, currentMeanF2); % Use {} for cell
        if ~isnan(currentMeanF2) && currentMeanF2 > bestF2Score
            bestF2Score = currentMeanF2;
            bestPipelineIdx = iPipeline;
        end
    else
        fprintf('Pipeline: %s results are missing or invalid for selection.\n', pipelines{iPipeline}.name); % Use {} for cell
    end
end

if bestPipelineIdx > 0
    bestPipelineSummary = allPipelinesResults{bestPipelineIdx};
    fprintf('\nBest Pipeline (Outlier Strategy: %s): %s with Mean F2_WHO3 = %.4f\n', ...
        outlierStrategy, bestPipelineSummary.pipelineConfig.name, bestF2Score); % Added strategy to log
else
    fprintf('\nNo suitable pipeline found or error in evaluation (Outlier Strategy: %s).\n', outlierStrategy); % Added strategy
    bestPipelineSummary = [];
end

% --- MODIFIED FILENAMES ---
resultsFilename = fullfile(resultsPath, sprintf('%s_Phase2_AllPipelineResults_Strat_%s.mat', dateStr, outlierStrategy));
save(resultsFilename, 'allPipelinesResults', 'pipelines', 'bestPipelineSummary', 'metricNames', 'numOuterFolds', 'numInnerFolds', 'outlierStrategy'); % Added outlierStrategy to saved vars
fprintf('All Phase 2 pipeline results (Outlier Strategy: %s) saved to: %s\n', outlierStrategy, resultsFilename);

if ~isempty(bestPipelineSummary)
    bestModelInfoFilename = fullfile(modelsPath, sprintf('%s_Phase2_BestPipelineInfo_Strat_%s.mat', dateStr, outlierStrategy));
    save(bestModelInfoFilename, 'bestPipelineSummary', 'outlierStrategy'); % Added outlierStrategy to saved vars
    fprintf('Best pipeline info (Outlier Strategy: %s) saved to: %s\n', outlierStrategy, bestModelInfoFilename);
end

%% 6. Visualization (Basic Example)
% =========================================================================
if ~isempty(allPipelinesResults) && bestPipelineIdx > 0 && ~isempty(f2_idx)
    try
        validPipelineIndices = find(cellfun(@(res) ~isempty(res) && isstruct(res) && isfield(res, 'outerFoldMetrics_mean') && length(res.outerFoldMetrics_mean) >= f2_idx, allPipelinesResults));
        if isempty(validPipelineIndices)
            fprintf('No valid pipeline results to plot (Outlier Strategy: %s).\n', outlierStrategy); % Added strategy
        else
            figCompare = figure('Name', sprintf('Pipeline Comparison (Strategy: %s) - F2 Score (WHO-3)', outlierStrategy), 'Position', [100, 100, 800, 600]); % Added strategy to figure name
            
            pipelineNamesToPlot = cellfun(@(p) p.name, pipelines(validPipelineIndices), 'UniformOutput', false);
            meanF2ScoresToPlot = cellfun(@(res) res.outerFoldMetrics_mean(f2_idx), allPipelinesResults(validPipelineIndices));
            stdF2ScoresToPlot = cellfun(@(res) res.outerFoldMetrics_std(f2_idx), allPipelinesResults(validPipelineIndices));

            bar(meanF2ScoresToPlot);
            hold on;
            errorbar(1:length(validPipelineIndices), meanF2ScoresToPlot, stdF2ScoresToPlot, 'k.', 'LineWidth', 1.5, 'HandleVisibility','off');
            
            xticks(1:length(validPipelineIndices));
            xticklabels(pipelineNamesToPlot);
            xtickangle(45);
            ylabel(sprintf('Mean %s', strrep(metricNames{f2_idx}, '_', ' ')));
            title(sprintf('Comparison of Pipelines (Nested CV Performance - Strategy: %s)', outlierStrategy), 'FontWeight','normal'); % Added strategy to title
            grid on;
            hold off;
            
            % --- MODIFIED FILENAME ---
            figFilenameBase = fullfile(figuresPath, sprintf('%s_Phase2_PipelineComparison_F2Score_Strat_%s', dateStr, outlierStrategy));
            savefig(figCompare, [figFilenameBase, '.fig']); % Use figCompare handle
            exportgraphics(figCompare, [figFilenameBase, '.tiff'], 'Resolution', 300); % Use figCompare handle
            fprintf('Pipeline comparison plot (Outlier Strategy: %s) saved to: %s.(fig/tiff)\n', outlierStrategy, figFilenameBase);
        end
    catch ME_plot
        fprintf('Error during plotting (Outlier Strategy: %s): %s\n', outlierStrategy, ME_plot.message); % Added strategy
        disp(ME_plot.getReport); 
    end
end

fprintf('\nPHASE 2 Processing Complete (Outlier Strategy: %s): %s\n', outlierStrategy, string(datetime('now'))); % Added strategy