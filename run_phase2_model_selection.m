function run_phase2_model_selection(cfg)
%RUN_PHASE2_MODEL_SELECTION
%
% Model and feature selection with optional outlier removal.
% Supports parallel evaluation of data with and without joint T2/Q
% outliers removed.
% Accepts either a configuration struct or a YAML file path.

if nargin < 1, cfg = struct(); end

helperPath = fullfile(fileparts(mfilename('fullpath')), 'helper_functions');
if exist('configure_cfg','file') ~= 2 && isfolder(helperPath)
    addpath(helperPath);
end

cfg = configure_cfg(cfg);
cfg = validate_configuration(cfg);

runConfig = load_run_configuration(cfg.projectRoot, cfg);
phase2Config = runConfig.phase2;
metricNames = phase2Config.metrics;
numOuterFolds = phase2Config.outerFolds;
numInnerFolds = phase2Config.innerFolds;
positiveClassLabel = runConfig.classLabels.positive;

% Add helper_functions/ to the path and obtain common directories
P = setup_project_paths(cfg.projectRoot,'Phase2');

logger = setup_logging(cfg, 'Phase2_ModelSelection');
loggerCleanup = onCleanup(@()logger.closeFcn()); %#ok<NASGU>
log_message('info', 'PHASE 2: Model Selection - %s', string(datetime('now')));
dataPath = P.dataPath;
resultsPathRoot = P.resultsPath;
modelsPathRoot = P.modelsPath;
if ~isfolder(resultsPathRoot); mkdir(resultsPathRoot); end
if ~isfolder(modelsPathRoot); mkdir(modelsPathRoot); end

% Configure random seed for reproducibility
phaseLogger = [];
if isfield(cfg,'logger'); phaseLogger = cfg.logger; end
[seedPhase2, seedSourcePhase2] = resolve_random_seed(cfg, 'randomSeedPhase2');
rngInfoPhase2 = set_random_seed(seedPhase2, 'Logger', phaseLogger, ...
    'Context', 'Phase 2 nested cross-validation');
rngMetadataBase = struct('phase','Phase2', ...
    'seedSource', seedSourcePhase2, ...
    'seedValueRequested', seedPhase2, ...
    'seedValueApplied', rngInfoPhase2.appliedSeed, ...
    'rngInfo', rngInfoPhase2);

%% Load base training data
trainData = load_dataset_split(dataPath, 'train');
wavenumbers_roi = trainData.wavenumbers;
X_all = trainData.X;
y_all = trainData.y;
probeIDs_all = trainData.probeIDs;

%% Build dataset variants
datasetVariants = create_dataset_variants(X_all, y_all, probeIDs_all, cfg);

if cfg.parallelOutlierComparison
    log_message('info', 'Running parallel comparison for %d dataset variants.', numel(datasetVariants));
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

datasetReporter = ProgressReporter('Phase 2 datasets', numel(datasetsToProcess), 'Verbose', cfg.verbose, 'ThrottleSeconds', 0);

%% Define pipelines
pipelines = define_pipelines();

%% Configure parallel execution
parEnv = get_parallel_environment_info();
if ~isfield(cfg,'enableParallelOuterCV') || isempty(cfg.enableParallelOuterCV)
    cfg.enableParallelOuterCV = parEnv.isAvailable;
end
useParallelOuter = logical(cfg.enableParallelOuterCV) && parEnv.isAvailable;
outerLogger = create_parallel_logger(useParallelOuter);
if useParallelOuter
    outerLogger('Parallel outer CV enabled (Parallel Computing Toolbox detected).');
else
    if ~parEnv.isAvailable
        fprintf('Parallel outer CV disabled: %s\n', parEnv.message);
    elseif ~cfg.enableParallelOuterCV
        fprintf('Parallel outer CV disabled via configuration flag.\n');
    else
        fprintf('Parallel outer CV disabled.\n');
    end
end

%% Run model selection for each dataset variant
for d = 1:numel(datasetsToProcess)
    ds = datasetsToProcess(d);
    log_message('info', '--- Dataset: %s ---', ds.description);

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
        numInnerFolds, resultsPath, modelsPath, 'Verbose', cfg.verbose);

    dateStr = string(datetime('now','Format','yyyyMMdd'));
    if cfg.parallelOutlierComparison
        resultsFile = fullfile(resultsPath, sprintf('%s_Phase2_%s_AllPipelineResults.mat', ...
            dateStr, ds.id));
    else
        resultsFile = fullfile(resultsPath, sprintf('%s_Phase2_AllPipelineResults.mat', dateStr));
    end

    metadata = rngMetadataBase;
    metadata.datasetID = ds.id;
    metadata.generatedOn = datetime('now');

    save(resultsFile,'resultsPerPipeline','pipelines','metricNames', ...
        'numOuterFolds','numInnerFolds','savedModels','ds');
    log_message('info', 'Results for %s saved to %s', ds.id, resultsFile);
end
end

%% ------------------------------------------------------------------------
function datasetVariants = create_dataset_variants(X_all, y_all, probeIDs_all, cfg)

    baseStruct = struct('id','FullData', ...
        'description','All spectra (no outlier removal)', ...
        'folderName','FullData', ...
        'X', X_all, ...
        'y', y_all, ...
        'probeIDs', probeIDs_all, ...
        'mask', true(size(y_all)), ...
        'outlierInfo', []);
    datasetVariants = baseStruct;

    try
        outlierStruct = identify_joint_t2q_outliers(X_all, cfg.outlierAlpha, cfg.outlierVarianceToModel);
        keepMask = outlierStruct.isJointInlier;
        if ~any(keepMask)
            log_message('warning', 'All spectra flagged as joint outliers. Filtered dataset skipped.');
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
            log_message('info', 'Joint T2/Q filtering removed %d/%d spectra.', ...
                outlierStruct.numJointOutliers, numel(keepMask));
        end
    catch ME
        log_message('warning', 'Failed to compute joint T2/Q outliers: %s', ME.message);
    end
end

function pipelinesOut = define_pipelines()
    pipelineList = cell(0,1); pidx = 0;

    pidx = pidx + 1;
    pipelineList{pidx} = pipelines.ClassificationPipeline( ...
        "BaselineLDA", ...
        pipelines.BinningTransformer(1), ...
        pipelines.NoFeatureSelector(), ...
        pipelines.LDAClassifier(), ...
        struct('binningFactor',[1 4 8]), ...
        {'binningFactor'});

    pidx = pidx + 1;
    pipelineList{pidx} = pipelines.ClassificationPipeline( ...
        "FisherLDA", ...
        pipelines.BinningTransformer(1), ...
        pipelines.FisherFeatureSelector(), ...
        pipelines.LDAClassifier(), ...
        struct('binningFactor',[4 8],'fisherFeaturePercent',[0.05 0.1 0.2 0.3 0.4 0.5]), ...
        {'binningFactor','fisherFeaturePercent'});

    pidx = pidx + 1;
    pipelineList{pidx} = pipelines.ClassificationPipeline( ...
        "PCALDA", ...
        pipelines.BinningTransformer(1), ...
        pipelines.PCAFeatureSelector(), ...
        pipelines.LDAClassifier(), ...
        struct('binningFactor',[4 8],'pcaVarianceToExplain',[0.90 0.95 0.99]), ...
        {'binningFactor','pcaVarianceToExplain'});

    pidx = pidx + 1;
    pipelineList{pidx} = pipelines.ClassificationPipeline( ...
        "MRMRLDA", ...
        pipelines.BinningTransformer(1), ...
        pipelines.MRMRFeatureSelector(), ...
        pipelines.LDAClassifier(), ...
        struct('binningFactor',[4 8],'mrmrFeaturePercent',[0.05 0.1 0.2 0.3 0.4]), ...
        {'binningFactor','mrmrFeaturePercent'});
    pipelinesOut = pipelineList;
end

function [resultsPerPipeline, savedModels] = perform_nested_cv_for_dataset(ds, pipelines, wavenumbers_roi, metricNames, numOuterFolds, numInnerFolds, resultsPath, modelsPath, varargin)

    p = inputParser();
    addParameter(p, 'Verbose', true, @(v) islogical(v) || isnumeric(v));
    parse(p, varargin{:});
    verbose = logical(p.Results.Verbose);

    X = ds.X; y = ds.y; probeIDs = ds.probeIDs;
    [uniqueProbes,~,groupIdx] = unique(probeIDs,'stable');
    if numel(uniqueProbes) < numOuterFolds
        error('Not enough unique probes (%d) for %d-fold CV in dataset %s.', ...
            numel(uniqueProbes), numOuterFolds, ds.id);
    end

    outerCV = cvpartition(length(uniqueProbes),'KFold',numOuterFolds);
    trainProbeIdx = cell(numOuterFolds,1);
    testProbeIdx = cell(numOuterFolds,1);
    for kFold = 1:numOuterFolds
        trainProbeIdx{kFold} = find(training(outerCV,kFold));
        testProbeIdx{kFold}  = find(test(outerCV,kFold));
    end

    resultsPerPipeline=cell(numel(pipelines),1);
    savedModels = cell(numel(pipelines),1);

    pipelineReporter = ProgressReporter(sprintf('Pipelines - %s', ds.id), numel(pipelines), 'Verbose', verbose, 'ThrottleSeconds', 0);

    for iPipe=1:numel(pipelines)
        pipe=pipelines{iPipe};
        fprintf('\nEvaluating pipeline: %s\n', char(pipe.Name));
        outerMetrics = NaN(numOuterFolds,numel(metricNames));
        outerBestHyper=cell(numOuterFolds,1);
        innerDiagnosticsPerFold = cell(numOuterFolds,1);
        trainingDiagnosticsPerFold = cell(numOuterFolds,1);
        for k=1:numOuterFolds
            trainIdx = training(outerCV,k);
            testIdx  = test(outerCV,k);
            trainMask = ismember(groupIdx, find(trainIdx));
            testMask  = ismember(groupIdx, find(testIdx));
            X_tr = X(trainMask,:); y_tr = y(trainMask); probes_tr = probeIDs(trainMask);
            X_te = X(testMask,:);  y_te = y(testMask);
            [bestHyper,~,innerDiagnostics] = perform_inner_cv(X_tr,y_tr,probes_tr,pipe,wavenumbers_roi,numInnerFolds,metricNames);
            innerDiagnosticsPerFold{k} = innerDiagnostics;
            if strcmpi(innerDiagnostics.status, 'error')
                log_pipeline_message('error', sprintf('run_phase2:innerCV:%s', pipe.name), ...
                    'Inner CV failed on outer fold %d. Marking fold as invalid.', k);
                outerBestHyper{k} = struct();
                continue;
            end

            outerBestHyper{k}=bestHyper;
            [finalModel,~,~, foldTrainDiagnostics] = train_final_pipeline_model(X_tr,y_tr,wavenumbers_roi,pipe,bestHyper);
            trainingDiagnosticsPerFold{k} = foldTrainDiagnostics;
            if strcmpi(foldTrainDiagnostics.status,'error') || ~isfield(finalModel,'LDAModel') || isempty(finalModel)
                log_pipeline_message('error', sprintf('run_phase2:train:%s', pipe.name), ...
                    'Failed to train outer fold %d model. Skipping metrics.', k);
                outerMetrics(k,:) = NaN;
                outerBestHyper{k} = struct();
                continue;
            end

            try
                [ypred,score] = apply_model_to_data(finalModel,X_te,wavenumbers_roi);
                posIdx=find(finalModel.LDAModel.ClassNames==3);
                m=calculate_performance_metrics(y_te,ypred,score(:,posIdx),3,metricNames);
                outerMetrics(k,:)=cell2mat(struct2cell(m))';
            catch ME_outer_eval
                log_pipeline_message('warning', sprintf('run_phase2:evaluation:%s', pipe.name), ...
                    'Evaluation failed on outer fold %d: %s', k, ME_outer_eval.message);
                outerMetrics(k,:) = NaN;
            end
        end
        res=struct();
        res.pipelineConfig=pipe;
        res.outerFoldMetrics_raw=outerMetrics;
        res.outerFoldMetrics_mean=nanmean(outerMetrics,1);
        res.outerFoldBestHyperparams=outerBestHyper;
        res.outerFoldDiagnostics = struct();
        res.outerFoldDiagnostics.inner = innerDiagnosticsPerFold;
        res.outerFoldDiagnostics.training = trainingDiagnosticsPerFold;

        aggHyper=aggregate_best_hyperparams(outerBestHyper);
        if isempty(fieldnames(aggHyper))
            log_pipeline_message('error', sprintf('run_phase2:aggregate:%s', pipe.name), ...
                'No valid hyperparameters aggregated. Skipping final model training.');
            finalModel = struct();
            selectedIdx = [];
            selectedWn = [];
            modelFile = '';
            finalDiagnostics = struct();
            finalDiagnostics.status = 'error';
            finalDiagnostics.entries = struct('timestamp',{},'level',{},'context',{},'message',{});
            finalDiagnostics.errors = {};
            finalDiagnostics.context = 'train_final_pipeline_model';
        else
            [finalModel,selectedIdx,selectedWn,finalDiagnostics]=train_final_pipeline_model(X,y,wavenumbers_roi,pipe,aggHyper);
            if strcmpi(finalDiagnostics.status,'error') || ~isfield(finalModel,'LDAModel') || isempty(finalModel)
                log_pipeline_message('error', sprintf('run_phase2:finalTrain:%s', pipe.name), ...
                    'Failed to train final model. Model will not be saved.');
                modelFile = '';
            else
                modelFile=fullfile(modelsPath,sprintf('%s_Phase2_%s_Model.mat',string(datetime('now','Format','yyyyMMdd')),pipe.name));
                save(modelFile,'finalModel','aggHyper','selectedIdx','selectedWn','ds','finalDiagnostics');
            end
        end
        res.finalModelFile=modelFile;
        res.finalModelDiagnostics = finalDiagnostics;

        resultsPerPipeline{iPipe}=res;
        savedModels{iPipe}=modelFile;
        pipelineReporter.update(1, sprintf('%s complete', pipe.name));
    end
end

