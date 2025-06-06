function run_phase3_final_evaluation_from_phase2(cfg)
%RUN_PHASE3_FINAL_EVALUATION_FROM_PHASE2
%
% Generic Phase 3 evaluation script that trains the final model using the
% best pipeline configuration saved from Phase 2. The script automatically
% loads the appropriate training data based on the outlier strategy stored
% in the Phase 2 file and evaluates the resulting MRMR-LDA model on the
% common test set.
%
% The required Phase 2 file is looked up in `models/Phase2` unless a
% specific path is provided in cfg.bestPipelineFile.
%
% Example:
%   cfg.bestPipelineFile = fullfile('models','Phase2','20250101_Phase2_BestPipelineInfo_Strat_AND.mat');
%   run('src/run_phase3_final_evaluation_from_phase2.m');
%
% Optional cfg fields:
%   projectRoot       - repository root (default: current dir)
%   bestPipelineFile  - path to *_Phase2_BestPipelineInfo_Strat_*.mat
%
% Date: 2025-06-07

%% 0. Initialization
fprintf('PHASE 3: Final Evaluation using Phase 2 output - %s\n', string(datetime('now')));

if nargin < 1
    cfg = struct();
end
if ~isfield(cfg, 'projectRoot')
    cfg.projectRoot = pwd;
end

% Setup project folders
P = setup_project_paths(cfg.projectRoot, 'Phase3');
dataPath         = P.dataPath;
phase2ModelsPath = fullfile(P.projectRoot, 'models', 'Phase2');
resultsPath      = P.resultsPath;
modelsPath       = P.modelsPath;
figuresPath      = P.figuresPath;

if ~exist(resultsPath, 'dir'), mkdir(resultsPath); end
if ~exist(modelsPath,  'dir'), mkdir(modelsPath); end
if ~exist(figuresPath, 'dir'), mkdir(figuresPath); end

% Locate the Phase 2 best pipeline file if not provided
if ~isfield(cfg, 'bestPipelineFile') || isempty(cfg.bestPipelineFile)
    files = dir(fullfile(phase2ModelsPath, '*_Phase2_BestPipelineInfo_Strat_*.mat'));
    if isempty(files)
        error('No Phase 2 best pipeline file found in %s.', phase2ModelsPath);
    end
    [~,idx] = sort([files.datenum],'descend');
    bestPipelineFile = fullfile(files(idx(1)).folder, files(idx(1)).name);
else
    bestPipelineFile = cfg.bestPipelineFile;
end
fprintf('Using Phase 2 best pipeline file: %s\n', bestPipelineFile);

% Load the file (expect variables bestPipelineSummary_strat and currentOutlierStrategy)
S = load(bestPipelineFile);
if isfield(S, 'bestPipelineSummary_strat')
    bestPipeline = S.bestPipelineSummary_strat;
elseif isfield(S, 'bestPipelineSummary')
    bestPipeline = S.bestPipelineSummary;
else
    error('Best pipeline summary not found in %s.', bestPipelineFile);
end
if isfield(S, 'currentOutlierStrategy')
    currentStrategy = upper(string(S.currentOutlierStrategy));
else
    % Fallback: try to parse from filename
    if contains(bestPipelineFile, 'Strat_OR', 'IgnoreCase', true)
        currentStrategy = "OR";
    else
        currentStrategy = "AND";
    end
end
fprintf('Outlier strategy from Phase 2: %s\n', currentStrategy);

% Determine training set and variable names based on strategy
switch upper(currentStrategy)
    case "OR"
        patternTrain = '*_training_set_no_outliers_T2orQ.mat';
        varName_X = 'X_train_no_outliers_OR';
        varName_y = 'y_train_no_outliers_OR_num';
        varName_probe = 'Patient_ID_no_outliers_OR';
    otherwise
        patternTrain = 'training_set_no_outliers_T2Q.mat';
        varName_X = 'X_train_no_outliers';
        varName_y = 'y_train_numeric_no_outliers';
        varName_probe = 'patientIDs_train_no_outliers';
end

filesTrain = dir(fullfile(dataPath, patternTrain));
if isempty(filesTrain)
    error('Training set for strategy %s not found using pattern %s in %s.', currentStrategy, patternTrain, dataPath);
end
[~,idxT] = sort([filesTrain.datenum],'descend');
trainFile = fullfile(filesTrain(idxT(1)).folder, filesTrain(idxT(1)).name);

fprintf('Loading training data from: %s\n', trainFile);
trainData = load(trainFile, varName_X, varName_y, varName_probe);
X_train_full = trainData.(varName_X);
y_train_full = trainData.(varName_y);

% Load wavenumbers
wData = load(fullfile(dataPath, 'wavenumbers.mat'), 'wavenumbers_roi');
wavenumbers_original = wData.wavenumbers_roi;
if iscolumn(wavenumbers_original), wavenumbers_original = wavenumbers_original'; end

% Load test data (common)
fprintf('Loading test set...\n');
T = load(fullfile(dataPath,'data_table_test.mat'), 'dataTableTest');
dataTableTest = T.dataTableTest;
numTestProbes = height(dataTableTest);
X_test_list  = cell(numTestProbes,1);
y_test_list  = cell(numTestProbes,1);
probe_test_list = cell(numTestProbes,1);
for i=1:numTestProbes
    M = dataTableTest.CombinedSpectra{i,1};
    N = size(M,1);
    X_test_list{i} = M;
    if dataTableTest.WHO_Grade(i) == 'WHO-1'
        y_test_list{i} = ones(N,1)*1;
    else
        y_test_list{i} = ones(N,1)*3;
    end
    probe_test_list{i} = repmat(dataTableTest.Diss_ID(i), N, 1);
end
X_test_full = vertcat(X_test_list{:});
y_test_full_numeric = vertcat(y_test_list{:});
probeIDs_test_full = vertcat(probe_test_list{:});

%% 1. Determine final hyperparameters from Phase 2 summary
final_binningFactor = 1; % default for MRMRLDA
final_numMRMRFeatures = 50; % default
if isfield(bestPipeline, 'pipelineConfig') && strcmpi(bestPipeline.pipelineConfig.name,'MRMRLDA')
    hp = bestPipeline.outerFoldBestHyperparams;
    numFeat = [];
    if iscell(hp)
        for k=1:numel(hp)
            if isfield(hp{k},'numMRMRFeatures')
                numFeat(end+1) = hp{k}.numMRMRFeatures; %#ok<AGROW>
            end
        end
    end
    if ~isempty(numFeat)
        final_numMRMRFeatures = mode(numFeat);
    end
else
    warning('Best pipeline from Phase 2 was %s, expected MRMRLDA. Default hyperparameters used.', bestPipeline.pipelineConfig.name);
end
fprintf('Using hyperparameters - Binning: %d, MRMR Features: %d\n', final_binningFactor, final_numMRMRFeatures);

metricNames = {'Accuracy','Sensitivity_WHO3','Specificity_WHO1','PPV_WHO3','NPV_WHO1','F1_WHO3','F2_WHO3','AUC'};

%% 2. Train Final Model
if final_binningFactor > 1
    [X_train_binned, wn_binned] = bin_spectra(X_train_full, wavenumbers_original, final_binningFactor);
else
    X_train_binned = X_train_full;
    wn_binned = wavenumbers_original;
end

X_train_fs = X_train_binned;
selected_idx = 1:size(X_train_binned,2);
if final_numMRMRFeatures > 0
    y_cat = categorical(y_train_full);
    ranked = fscmrmr(X_train_binned, y_cat);
    nsel = min(final_numMRMRFeatures, numel(ranked));
    selected_idx = ranked(1:nsel);
    X_train_fs = X_train_binned(:, selected_idx);
end
final_LDA = fitcdiscr(X_train_fs, y_train_full);

%% 3. Evaluate on Test Set
if final_binningFactor > 1
    [X_test_binned, ~] = bin_spectra(X_test_full, wavenumbers_original, final_binningFactor);
else
    X_test_binned = X_test_full;
end
X_test_fs = X_test_binned(:, selected_idx);
[y_pred_test, y_scores_test] = predict(final_LDA, X_test_fs);
posClass = 3;
classCol = find(final_LDA.ClassNames == posClass);
score_pos = y_scores_test(:, classCol);

metricsSpectrum = calculate_performance_metrics(y_test_full_numeric, y_pred_test, score_pos, posClass, metricNames);

%% 4. Aggregate Probe-Level
uniqueProbes = unique(probeIDs_test_full, 'stable');
probeLevelResults = table();
probeLevelResults.Diss_ID = uniqueProbes;
probeLevelResults.True_WHO_Grade_Numeric = NaN(numel(uniqueProbes),1);
probeLevelResults.Mean_WHO3_Probability = NaN(numel(uniqueProbes),1);
probeLevelResults.Predicted_WHO_Grade_Numeric_MeanProb = NaN(numel(uniqueProbes),1);

for i=1:numel(uniqueProbes)
    idx = strcmp(probeIDs_test_full, uniqueProbes{i});
    probeLevelResults.True_WHO_Grade_Numeric(i) = mode(y_test_full_numeric(idx));
    probeLevelResults.Mean_WHO3_Probability(i) = mean(score_pos(idx));
    probeLevelResults.Predicted_WHO_Grade_Numeric_MeanProb(i) = probeLevelResults.Mean_WHO3_Probability(i) > 0.5 .* 3 + ...
        (probeLevelResults.Mean_WHO3_Probability(i) <= 0.5) .* 1;
end

metricsProbe = calculate_performance_metrics(probeLevelResults.True_WHO_Grade_Numeric, ...
    probeLevelResults.Predicted_WHO_Grade_Numeric_MeanProb, probeLevelResults.Mean_WHO3_Probability, posClass, metricNames);

%% 5. Save Model and Results
dateStr = string(datetime('now','Format','yyyyMMdd'));
modelPackage = struct();
modelPackage.description = 'Final MRMRLDA model trained using best Phase 2 output';
modelPackage.trainingDate = string(datetime('now'));
modelPackage.LDAModel = final_LDA;
modelPackage.binningFactor = final_binningFactor;
modelPackage.numMRMRFeaturesSelected = numel(selected_idx);
modelPackage.selectedFeatureIndices_in_binned_space = selected_idx;
modelPackage.selectedWavenumbers = wn_binned(selected_idx);
modelPackage.originalWavenumbers_before_binning = wavenumbers_original;
modelPackage.binnedWavenumbers_for_selection = wn_binned;
modelPackage.testSetPerformance_Spectrum = metricsSpectrum;
modelPackage.probeLevelResults = probeLevelResults;
modelPackage.probeLevelPerformance_MeanProb = metricsProbe;
modelPackage.trainingDataFile = trainFile;
modelPackage.testDataFile = fullfile(dataPath,'data_table_test.mat');
modelPackage.outlierStrategyUsed = char(currentStrategy);

modelFilename = fullfile(modelsPath, sprintf('%s_Phase3_FinalModel_FromPhase2.mat', dateStr));
save(modelFilename, 'modelPackage');
resultsFilename = fullfile(resultsPath, sprintf('%s_Phase3_TestResults_FromPhase2.mat', dateStr));
save(resultsFilename, 'metricsSpectrum', 'probeLevelResults', 'metricsProbe', 'final_binningFactor', 'final_numMRMRFeatures');

fprintf('Saved model to %s\n', modelFilename);
fprintf('Saved results to %s\n', resultsFilename);

fprintf('PHASE 3 complete.\n');
end
