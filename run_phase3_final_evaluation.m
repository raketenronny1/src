function run_phase3_final_evaluation(cfg)
%RUN_PHASE3_FINAL_EVALUATION
%
% Apply each pipeline model saved in Phase 2 to the test data and compare
% performance for different outlier-handling strategies. When Phase 2 was
% executed with `parallelOutlierComparison`, the script evaluates the
% resulting model sets separately and reports results for test data with
% and without joint Hotelling T2 / Q-statistic outliers.
% Accepts either a configuration struct or a YAML file path.

%% 0. Configuration
if nargin < 1 || isempty(cfg)
    cfg = configure_cfg();
elseif ischar(cfg) || (isstring(cfg) && isscalar(cfg))
    cfg = configure_cfg('configFile', char(cfg));
elseif ~isstruct(cfg)
    error('run_phase3_final_evaluation:InvalidConfig', ...
        'Configuration input must be empty, a struct or a file path.');
end

helperPath = fullfile(fileparts(mfilename('fullpath')), 'helper_functions');
if exist('configure_cfg','file') ~= 2 && isfolder(helperPath)
    addpath(helperPath);
end

cfg = configure_cfg(cfg);
cfg = validate_configuration(cfg);

runConfig = load_run_configuration(cfg.projectRoot, cfg);
phase3Config = runConfig.phase3;
metricNamesEval = phase3Config.metrics;
probeMetricNames = phase3Config.probeMetrics;
positiveClassLabel = runConfig.classLabels.positive;
negativeClassLabel = runConfig.classLabels.negative;

P = setup_project_paths(cfg.projectRoot,'Phase3',cfg);
resultsPath = P.resultsPath;
figuresPath = P.figuresPath;
modelsPathP2 = fullfile(cfg.projectRoot,'models','Phase2');
resultsPathP2 = fullfile(cfg.projectRoot,'results','Phase2');
if ~isfolder(resultsPath); mkdir(resultsPath); end
if ~isfolder(figuresPath); mkdir(figuresPath); end

logger = setup_logging(cfg, 'Phase3_FinalEvaluation');
loggerCleanup = onCleanup(@()logger.closeFcn()); %#ok<NASGU>
log_message('info', 'PHASE 3: Final Evaluation - %s', string(datetime('now')));

%% 1. Load test data and build evaluation variants
log_message('info', 'Loading test set...');
dataPath = P.dataPath;
testData = load_dataset_split(dataPath, 'test');
wavenumbers = testData.wavenumbers;
X_test = testData.X;
y_test = testData.y;
probeIDs_test = testData.probeIDs;

testVariants = build_test_variants(X_test, y_test, probeIDs_test, cfg);

%% 2. Locate model sets and Phase 2 results
modelSets = discover_model_sets(modelsPathP2, resultsPathP2);
if isempty(modelSets)
    error('No Phase 2 models found in %s.', modelsPathP2);
end

%% 3. Evaluate models across variants
resultsByVariant = struct('id',{},'description',{},'modelSets',{});
bestModelInfo = struct('variantID',{},'modelSetID',{},'modelName',{},'metrics',{},'modelFile',{});

variantReporter = ProgressReporter('Phase 3 variants', numel(testVariants), 'Verbose', cfg.verbose, 'ThrottleSeconds', 0);

for v = 1:numel(testVariants)
    variant = testVariants(v);
    log_message('info', 'Evaluating test variant: %s', variant.description);
    variantResults = struct('modelSetID',{},'modelSetDescription',{},'models',{});
    bestScore = -Inf; bestEntry = struct();
    modelSetReporter = ProgressReporter(sprintf('Model sets - %s', variant.id), numel(modelSets), 'Verbose', cfg.verbose, 'ThrottleSeconds', 0);

    for s = 1:numel(modelSets)
        modelSet = modelSets(s);
        fprintf('  Model set: %s\n', modelSet.description);
        models = evaluate_model_set(modelSet, variant, wavenumbers, metricNamesEval, figuresPath, positiveClassLabel, negativeClassLabel, probeMetricNames);
        variantResults(end+1).modelSetID = modelSet.id; %#ok<AGROW>
        variantResults(end).modelSetDescription = modelSet.description;
        variantResults(end).models = models;

        % Track best model for this variant based on F2_WHO3
        for mIdx = 1:numel(models)
            if isfield(models(mIdx).metrics,'F2_WHO3') && models(mIdx).metrics.F2_WHO3 > bestScore
                bestScore = models(mIdx).metrics.F2_WHO3;
                bestEntry.variantID = variant.id;
                bestEntry.modelSetID = modelSet.id;
                bestEntry.modelName = models(mIdx).name;
                bestEntry.metrics = models(mIdx).metrics;
                bestEntry.modelFile = models(mIdx).modelFile;
            end
        end
        modelSetReporter.update(1, sprintf('%s complete', modelSet.id));
    end

    resultsByVariant(v).id = variant.id; %#ok<AGROW>
    resultsByVariant(v).description = variant.description;
    resultsByVariant(v).modelSets = variantResults;
    if ~isempty(fieldnames(bestEntry))
        bestModelInfo(v) = bestEntry; %#ok<AGROW>
    end
    variantReporter.update(1, sprintf('%s complete', variant.id));
    if cfg.verbose
        fprintf('Completed evaluations for variant %s.\n', variant.id);
    end
end

%% Save combined results
dateStr = string(datetime('now','Format','yyyyMMdd'));
resultsFile = fullfile(resultsPath,sprintf('%s_Phase3_ParallelComparisonResults.mat',dateStr));
save(resultsFile,'resultsByVariant','bestModelInfo','testVariants','modelSets');
log_message('info', 'Saved Phase 3 comparison results to %s', resultsFile);

end

%% Helper functions
function testVariants = build_test_variants(X_test, y_test, probeIDs_test, cfg)

    testVariants = struct('id',{},'description',{},'X',{},'y',{},'probeIDs',{},'mask',{},'outlierInfo',{});
    baseVariant = struct('id','FullTest', ...
        'description','All test spectra (no outlier removal)', ...
        'X', X_test, ...
        'y', y_test, ...
        'probeIDs', probeIDs_test, ...
        'mask', true(size(y_test)), ...
        'outlierInfo', []);
    testVariants(end+1) = baseVariant; %#ok<AGROW>

    try
        outlierStruct = identify_joint_t2q_outliers(X_test, cfg.outlierAlpha, cfg.outlierVarianceToModel);
        keepMask = outlierStruct.isJointInlier;
        if any(keepMask) && any(outlierStruct.isJointOutlier)
            filteredVariant = baseVariant;
            filteredVariant.id = 'FilteredTest';
            filteredVariant.description = sprintf('Test spectra without joint T2/Q outliers (%d removed)', ...
                outlierStruct.numJointOutliers);
            filteredVariant.X = X_test(keepMask,:);
            filteredVariant.y = y_test(keepMask);
            filteredVariant.probeIDs = probeIDs_test(keepMask);
            filteredVariant.mask = keepMask;
            filteredVariant.outlierInfo = outlierStruct;
            testVariants(end+1) = filteredVariant; %#ok<AGROW>
            log_message('info', 'Joint T2/Q filtering removed %d/%d test spectra.', ...
                outlierStruct.numJointOutliers, numel(keepMask));
        end
    catch ME
        log_message('warning', 'Failed to compute joint outliers on test set: %s', ME.message);
    end
end

function [seedValue, sourceField] = resolve_phase3_seed(cfg)

    seedValue = [];
    sourceField = '';

    if isfield(cfg,'randomSeedPhase3') && ~isempty(cfg.randomSeedPhase3)
        seedValue = cfg.randomSeedPhase3;
        sourceField = 'randomSeedPhase3';
        return;
    end

    if isfield(cfg,'randomSeed') && ~isempty(cfg.randomSeed)
        seedValue = cfg.randomSeed;
        sourceField = 'randomSeed';
    end
end

function modelSets = discover_model_sets(modelsPathP2, resultsPathP2)

    modelSets = struct('id',{},'description',{},'modelsDir',{},'resultsDir',{},'modelFiles',{},'cvData',{},'pipelines',{},'metricNames',{});

    % Helper to register a model directory
    function addModelSet(setID, setDescription, modelDir, resultsDir)
        files = dir(fullfile(modelDir, '*_Phase2_*_Model.mat'));
        if isempty(files)
            return;
        end
        files = select_latest_models(files);
        cvInfo = load_cv_results(resultsDir);
        modelSets(end+1) = struct( ...
            'id', setID, ...
            'description', setDescription, ...
            'modelsDir', modelDir, ...
            'resultsDir', resultsDir, ...
            'modelFiles', files, ...
            'cvData', cvInfo.cvData, ...
            'pipelines', cvInfo.pipelines, ...
            'metricNames', cvInfo.metricNames); %#ok<AGROW>
    end

    % Root directory (legacy single-run scenario)
    addModelSet('Default','Models (root Phase2 directory)', modelsPathP2, resultsPathP2);

    % Subdirectories for parallel comparisons
    dirInfo = dir(modelsPathP2);
    for i = 1:numel(dirInfo)
        if dirInfo(i).isdir && ~ismember(dirInfo(i).name,{'.','..'})
            subdir = fullfile(modelsPathP2, dirInfo(i).name);
            addModelSet(dirInfo(i).name, sprintf('Models - %s', dirInfo(i).name), subdir, fullfile(resultsPathP2, dirInfo(i).name));
        end
    end

    % Remove duplicates if both root and subdir have identical IDs with no files
    modelSets = modelSets(~arrayfun(@(s) isempty(s.modelFiles), modelSets));
end

function files = select_latest_models(modelFiles)

    fileMap = containers.Map();
    for i = 1:numel(modelFiles)
        tokens = regexp(modelFiles(i).name,'^\d+_Phase2_(.+)_Model\.mat$','tokens','once');
        if isempty(tokens); continue; end
        pipe = tokens{1};
        if ~isKey(fileMap, pipe) || modelFiles(i).datenum > fileMap(pipe).datenum
            fileMap(pipe) = modelFiles(i);
        end
    end
    filesCell = values(fileMap);
    files = [filesCell{:}];
    [~,order] = sort({files.name});
    files = files(order);
end

function cvInfo = load_cv_results(resultsDir)

    cvInfo = struct('cvData',[],'pipelines',[],'metricNames',[]);
    if ~isfolder(resultsDir)
        return;
    end
    resFile = dir(fullfile(resultsDir,'*_Phase2_*_AllPipelineResults.mat'));
    if isempty(resFile)
        return;
    end
    [~,idx] = sort([resFile.datenum],'descend');
    tmp = load(fullfile(resFile(idx(1)).folder,resFile(idx(1)).name));
    if isfield(tmp,'resultsPerPipeline'); cvInfo.cvData = tmp.resultsPerPipeline; end
    if isfield(tmp,'pipelines'); cvInfo.pipelines = tmp.pipelines; end
    if isfield(tmp,'metricNames'); cvInfo.metricNames = tmp.metricNames; end
end

function models = evaluate_model_set(modelSet, variant, wavenumbers, metricNamesEval, figuresPath, positiveClassLabel, negativeClassLabel, probeMetricNames, varargin)

    models = struct('name',{},'metrics',{},'modelFile',{},'scores',{},'predicted',{},'probeTable',{},'probeMetrics',{},'CV_Metrics',{},'rocFile',{});
    X = variant.X;
    y = variant.y;
    probeIDs = variant.probeIDs;

    pipeline_names_from_cv = {};
    if ~isempty(modelSet.pipelines)
        pipeline_names_from_cv = cellfun(@extract_pipeline_name, modelSet.pipelines, 'UniformOutput', false);
    end

    p = inputParser();
    addParameter(p, 'Verbose', true, @(v) islogical(v) || isnumeric(v));
    parse(p, varargin{:});
    verbose = logical(p.Results.Verbose);

    modelReporter = ProgressReporter(sprintf('Models - %s | %s', modelSet.id, variant.id), numel(modelSet.modelFiles), 'Verbose', verbose, 'ThrottleSeconds', 0);

    for i=1:numel(modelSet.modelFiles)
        mf = fullfile(modelSet.modelFiles(i).folder,modelSet.modelFiles(i).name);
        S = load(mf,'finalModel','aggHyper','selectedIdx','selectedWn','ds'); %#ok<NASGU>
        if ~isfield(S,'finalModel')
            log_message('warning', 'Model file %s missing finalModel. Skipping.', mf);
            continue;
        end
        finalModel = S.finalModel;
        if isa(finalModel,'pipelines.TrainedClassificationPipeline')
            mdlName = char(finalModel.Name);
        elseif isfield(finalModel,'pipelineName')
            mdlName = finalModel.pipelineName;
        elseif isfield(finalModel,'featureSelectionMethod')
            mdlName = finalModel.featureSelectionMethod;
        else
            mdlName = sprintf('Model_%d', i);
        end
        mdlName = char(mdlName);

        [ypred,score] = apply_model_to_data(finalModel,X,wavenumbers);
        [metrics, posScores] = evaluate_pipeline_metrics(y, ypred, score, finalModel.LDAModel.ClassNames, metricNamesEval, positiveClassLabel);
        if isempty(posScores) && ~isempty(score)
            posIdx = find_positive_class_index(finalModel.LDAModel.ClassNames, positiveClassLabel);
            if ~isempty(posIdx)
                posScores = score(:, posIdx);
            end
        end

        entry = struct();
        entry.name = mdlName;
        entry.metrics = metrics;
        entry.modelFile = mf;
        entry.scores = posScores;
        entry.predicted = ypred;

        [probeTable,probeMetrics] = aggregate_probe_metrics(probeIDs, y, posScores, ypred, metricNamesEval, positiveClassLabel, negativeClassLabel, probeMetricNames);
        entry.probeTable = probeTable;
        entry.probeMetrics = probeMetrics;

        % Attach CV metrics if available
        if ~isempty(modelSet.cvData)
            idx = find(strcmpi(pipeline_names_from_cv, entry.name),1);
            if ~isempty(idx) && numel(modelSet.cvData) >= idx
                entry.CV_Metrics = modelSet.cvData{idx}.outerFoldMetrics_mean;
            end
        end

        % ROC curve file per variant/model set combination
        if isempty(posScores)
            Xroc = [0 1]; Yroc = [0 1]; AUC = NaN;
        else
            [Xroc,Yroc,~,AUC] = perfcurve(y,posScores,positiveClassLabel);
        end
        rocFile = fullfile(figuresPath,sprintf('ROC_%s_%s_%s.png', entry.name, modelSet.id, variant.id));
        fig = figure('Visible','off');
        plot(Xroc,Yroc,'LineWidth',1.5); grid on;
        xlabel('False positive rate'); ylabel('True positive rate');
        title(sprintf('ROC - %s (%s, %s) AUC %.3f',entry.name,modelSet.id,variant.id,AUC));
        saveas(fig,rocFile); close(fig);
        entry.rocFile = rocFile;

        models(end+1) = entry; %#ok<AGROW>
        modelReporter.update(1, sprintf('%s complete', entry.name));
    end
end

function [tbl,metrics] = aggregate_probe_metrics(probeIDs, yTrue, scores, yPred, metricNames, positiveClassLabel, negativeClassLabel, probeMetricNames)

    if nargin < 8 || isempty(probeMetricNames)
        probeMetricNames = metricNames;
    end
    if nargin < 6 || isempty(positiveClassLabel)
        positiveClassLabel = 3;
    end

    if nargin < 7 || isempty(negativeClassLabel)
        if isnumeric(positiveClassLabel) || islogical(positiveClassLabel)
            negativeClassLabel = 0;
        elseif isstring(positiveClassLabel)
            negativeClassLabel = string(missing);
        elseif ischar(positiveClassLabel)
            negativeClassLabel = '';
        elseif iscategorical(positiveClassLabel)
            negativeClassLabel = categorical(missing);
        else
            negativeClassLabel = positiveClassLabel;
        end
    end

    % probeIDs should be an array of probe identifiers (numeric or string).
    probeIDs = string(probeIDs); % ensure string comparison
    probes = unique(probeIDs,'stable');

    numProbes = numel(probes);
    tbl = table();
    tbl.Diss_ID = probes;
    tbl.MeanProbWHO3 = NaN(numProbes,1);

    trueTemplate = determine_label_template(yTrue, positiveClassLabel, negativeClassLabel);
    predTemplate = determine_label_template(yPred, negativeClassLabel, positiveClassLabel);
    tbl.TrueLabel = repmat(trueTemplate, numProbes, 1);
    tbl.PredLabel = repmat(predTemplate, numProbes, 1);

    for i = 1:numProbes
        idx = strcmp(probeIDs, probes(i));
        if ~isempty(yTrue)
            tbl.TrueLabel(i) = mode(yTrue(idx));
        end

        if isempty(scores)
            tbl.MeanProbWHO3(i) = NaN;
        else
            tbl.MeanProbWHO3(i) = mean(scores(idx));
        end

        if ~isnan(tbl.MeanProbWHO3(i))
            if tbl.MeanProbWHO3(i) >= 0.5
                tbl.PredLabel(i) = positiveClassLabel;
            else
                tbl.PredLabel(i) = negativeClassLabel;
            end
        elseif ~isempty(yPred)
            tbl.PredLabel(i) = mode(yPred(idx));
        else
            tbl.PredLabel(i) = negativeClassLabel;
        end
    end

    metricList = probeMetricNames;
    if isempty(metricList)
        metricList = metricNames;
    end

    metrics = evaluate_pipeline_metrics(tbl.TrueLabel, tbl.PredLabel, tbl.MeanProbWHO3, [], metricList, positiveClassLabel);
end

function idx = find_positive_class_index(classNames, positiveClassLabel)
    if isa(classNames, 'categorical')
        classNames = string(classNames);
    end
    if iscell(classNames)
        classNames = string(classNames);
    end

    if isnumeric(classNames)
        idx = find(classNames == positiveClassLabel, 1, 'first');
        return;
    end

    classStr = string(classNames);
    posStr = string(positiveClassLabel);
    idx = find(classStr == posStr, 1, 'first');
    if isempty(idx) && all(~isnan(str2double(classStr)))
        numericClasses = str2double(classStr);
        idx = find(numericClasses == positiveClassLabel, 1, 'first');
    end
end

function template = determine_label_template(values, primaryFallback, secondaryFallback)
    if nargin < 2
        primaryFallback = [];
    end
    if nargin < 3
        secondaryFallback = [];
    end

    if ~isempty(values)
        template = values(1);
        return;
    end

    if ~isempty(primaryFallback)
        template = primaryFallback;
        return;
    end

    if ~isempty(secondaryFallback)
        template = secondaryFallback;
        return;
    end

    template = 0;
end
