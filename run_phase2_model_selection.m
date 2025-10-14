function run_phase2_model_selection(cfg)
%RUN_PHASE2_MODEL_SELECTION
%
% Model and feature selection with optional outlier removal.
% Supports parallel evaluation of data with and without joint T2/Q
% outliers removed.

fprintf('PHASE 2: Model Selection - %s\n', string(datetime('now')));
if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'projectRoot'); cfg.projectRoot = pwd; end

% Add helper_functions/ to the path and obtain common directories
P = setup_project_paths(cfg.projectRoot,'Phase2');
dataPath = P.dataPath;
resultsPathRoot = P.resultsPath;
modelsPathRoot = P.modelsPath;
if ~isfolder(resultsPathRoot); mkdir(resultsPathRoot); end
if ~isfolder(modelsPathRoot); mkdir(modelsPathRoot); end

%% Load base training data
trainTablePath = fullfile(dataPath,'data_table_train.mat');
if ~isfile(trainTablePath)
    error('Training table not found: %s', trainTablePath);
end
load(trainTablePath,'dataTableTrain');
load(fullfile(dataPath,'wavenumbers.mat'),'wavenumbers_roi');
if iscolumn(wavenumbers_roi); wavenumbers_roi = wavenumbers_roi'; end

[X_all, y_all, ~, probeIDs_all] = flatten_spectra_for_pca( ...
    dataTableTrain, length(wavenumbers_roi));

%% Build dataset variants
datasetVariants = create_dataset_variants(X_all, y_all, probeIDs_all, cfg);

if cfg.parallelOutlierComparison
    fprintf('Running parallel comparison for %d dataset variants.\n', numel(datasetVariants));
    datasetsToProcess = datasetVariants;
else
    % Select variant based on legacy flag for backwards compatibility
    if isfield(cfg,'useOutlierRemoval') && cfg.useOutlierRemoval
        variantIdx = find(strcmp({datasetVariants.id}, 'FilteredT2Q'), 1);
        if isempty(variantIdx)
            error('Filtered dataset variant not available.');
        end
    else
        variantIdx = find(strcmp({datasetVariants.id}, 'FullData'), 1);
        if isempty(variantIdx)
            error('Full dataset variant not available.');
        end
    end
    datasetsToProcess = datasetVariants(variantIdx);
end

%% Define pipelines
pipelines = define_pipelines();
metricNames = {'Accuracy','Sensitivity_WHO3','Specificity_WHO1', ...
    'PPV_WHO3','NPV_WHO1','F1_WHO3','F2_WHO3','AUC'};
numOuterFolds = 5; numInnerFolds = 3;

%% Run model selection for each dataset variant
for d = 1:numel(datasetsToProcess)
    ds = datasetsToProcess(d);
    fprintf('\n--- Dataset: %s ---\n', ds.description);

    if cfg.parallelOutlierComparison
        resultsPath = fullfile(resultsPathRoot, ds.folderName);
        modelsPath = fullfile(modelsPathRoot, ds.folderName);
    else
        resultsPath = resultsPathRoot;
        modelsPath = modelsPathRoot;
    end
    if ~isfolder(resultsPath); mkdir(resultsPath); end
    if ~isfolder(modelsPath); mkdir(modelsPath); end

    [resultsPerPipeline, savedModels] = perform_nested_cv_for_dataset( ...
        ds, pipelines, wavenumbers_roi, metricNames, numOuterFolds, ...
        numInnerFolds, resultsPath, modelsPath);

    dateStr = string(datetime('now','Format','yyyyMMdd'));
    if cfg.parallelOutlierComparison
        resultsFile = fullfile(resultsPath, sprintf('%s_Phase2_%s_AllPipelineResults.mat', ...
            dateStr, ds.id));
    else
        resultsFile = fullfile(resultsPath, sprintf('%s_Phase2_AllPipelineResults.mat', dateStr));
    end

    save(resultsFile,'resultsPerPipeline','pipelines','metricNames', ...
        'numOuterFolds','numInnerFolds','savedModels','ds');
    fprintf('Results for %s saved to %s\n', ds.id, resultsFile);
end
end

%% ------------------------------------------------------------------------
function datasetVariants = create_dataset_variants(X_all, y_all, probeIDs_all, cfg)

    datasetVariants = struct('id',{},'description',{},'folderName',{}, ...
        'X',{},'y',{},'probeIDs',{},'mask',{},'outlierInfo',{});

    baseStruct = struct('id','FullData', ...
        'description','All spectra (no outlier removal)', ...
        'folderName','FullData', ...
        'X', X_all, ...
        'y', y_all, ...
        'probeIDs', probeIDs_all, ...
        'mask', true(size(y_all)), ...
        'outlierInfo', []);
    datasetVariants(end+1) = baseStruct; %#ok<AGROW>

    try
        outlierStruct = identify_joint_t2q_outliers(X_all, cfg.outlierAlpha, cfg.outlierVarianceToModel);
        keepMask = outlierStruct.isJointInlier;
        if ~any(keepMask)
            warning('All spectra flagged as joint outliers. Filtered dataset skipped.');
        else
            filteredStruct = baseStruct;
            filteredStruct.id = 'FilteredT2Q';
            filteredStruct.description = sprintf('Joint T2/Q filtered (removed %d of %d spectra)', ...
                outlierStruct.numJointOutliers, numel(keepMask));
            filteredStruct.folderName = 'FilteredT2Q';
            filteredStruct.X = X_all(keepMask,:);
            filteredStruct.y = y_all(keepMask);
            filteredStruct.probeIDs = probeIDs_all(keepMask);
            filteredStruct.mask = keepMask;
            filteredStruct.outlierInfo = outlierStruct;
            datasetVariants(end+1) = filteredStruct; %#ok<AGROW>
            fprintf('Joint T2/Q filtering removed %d/%d spectra.\n', ...
                outlierStruct.numJointOutliers, numel(keepMask));
        end
    catch ME
        warning('Failed to compute joint T2/Q outliers: %s', ME.message);
    end
end

function pipelines = define_pipelines()
    pipelines = cell(0,1); pidx=0;

    p=struct(); p.name='BaselineLDA'; p.feature_selection_method='none'; p.classifier='LDA';
    p.hyperparameters_to_tune={'binningFactor'}; p.binningFactors=[4 8];
    pidx=pidx+1; pipelines{pidx}=p;

    p=struct(); p.name='FisherLDA'; p.feature_selection_method='fisher'; p.classifier='LDA';
    p.hyperparameters_to_tune={'binningFactor','fisherFeaturePercent'};
    p.binningFactors=[4 8]; p.fisherFeaturePercent_range=[0.05 0.1 0.2 0.3 0.4 0.5];
    pidx=pidx+1; pipelines{pidx}=p;

    p=struct(); p.name='PCALDA'; p.feature_selection_method='pca'; p.classifier='LDA';
    p.hyperparameters_to_tune={'binningFactor','pcaVarianceToExplain'};
    p.binningFactors=[4 8]; p.pcaVarianceToExplain_range=[0.90 0.95 0.99];
    pidx=pidx+1; pipelines{pidx}=p;

    p=struct(); p.name='MRMRLDA'; p.feature_selection_method='mrmr'; p.classifier='LDA';
    p.hyperparameters_to_tune={'binningFactor','mrmrFeaturePercent'};
    p.binningFactors=[4 8]; p.mrmrFeaturePercent_range=[0.05 0.1 0.2 0.3 0.4];
    pidx=pidx+1; pipelines{pidx}=p;
end

function [resultsPerPipeline, savedModels] = perform_nested_cv_for_dataset(ds, pipelines, wavenumbers_roi, metricNames, numOuterFolds, numInnerFolds, resultsPath, modelsPath)

    X = ds.X; y = ds.y; probeIDs = ds.probeIDs;
    [uniqueProbes,~,groupIdx] = unique(probeIDs,'stable');
    if numel(uniqueProbes) < numOuterFolds
        error('Not enough unique probes (%d) for %d-fold CV in dataset %s.', ...
            numel(uniqueProbes), numOuterFolds, ds.id);
    end

    outerCV = cvpartition(length(uniqueProbes),'KFold',numOuterFolds);
    resultsPerPipeline=cell(numel(pipelines),1);
    savedModels = cell(numel(pipelines),1);

    for iPipe=1:numel(pipelines)
        pipe=pipelines{iPipe};
        fprintf('\nEvaluating pipeline: %s\n', pipe.name);
        outerMetrics = NaN(numOuterFolds,numel(metricNames));
        outerBestHyper=cell(numOuterFolds,1);
        for k=1:numOuterFolds
            trainIdx = training(outerCV,k);
            testIdx  = test(outerCV,k);
            trainMask = ismember(groupIdx, find(trainIdx));
            testMask  = ismember(groupIdx, find(testIdx));
            X_tr = X(trainMask,:); y_tr = y(trainMask); probes_tr = probeIDs(trainMask);
            X_te = X(testMask,:);  y_te = y(testMask);
            [bestHyper,~] = perform_inner_cv(X_tr,y_tr,probes_tr,pipe,wavenumbers_roi,numInnerFolds,metricNames);
            outerBestHyper{k}=bestHyper;
            [finalModel,~,~] = train_final_pipeline_model(X_tr,y_tr,wavenumbers_roi,pipe,bestHyper);
            [ypred,score] = apply_model_to_data(finalModel,X_te,wavenumbers_roi);
            posIdx=find(finalModel.LDAModel.ClassNames==3);
            m=calculate_performance_metrics(y_te,ypred,score(:,posIdx),3,metricNames);
            outerMetrics(k,:)=cell2mat(struct2cell(m))';
        end
        res=struct();
        res.pipelineConfig=pipe;
        res.outerFoldMetrics_raw=outerMetrics;
        res.outerFoldMetrics_mean=nanmean(outerMetrics,1);
        res.outerFoldBestHyperparams=outerBestHyper;

        aggHyper=aggregate_best_hyperparams(outerBestHyper);
        [finalModel,selectedIdx,selectedWn]=train_final_pipeline_model(X,y,wavenumbers_roi,pipe,aggHyper);
        modelFile=fullfile(modelsPath,sprintf('%s_Phase2_%s_Model.mat',string(datetime('now','Format','yyyyMMdd')),pipe.name));
        save(modelFile,'finalModel','aggHyper','selectedIdx','selectedWn','ds');
        res.finalModelFile=modelFile;

        resultsPerPipeline{iPipe}=res;
        savedModels{iPipe}=modelFile;
    end
end
