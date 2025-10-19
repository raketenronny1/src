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
if ~isfield(cfg,'projectRoot'); cfg.projectRoot = pwd; end
if ~isfield(cfg,'outlierAlpha'); cfg.outlierAlpha = 0.01; end
if ~isfield(cfg,'outlierVarianceToModel'); cfg.outlierVarianceToModel = 0.95; end

P = setup_project_paths(cfg.projectRoot,'Phase3',cfg);
resultsPath = P.resultsPath;
figuresPath = P.figuresPath;
modelsPathP2 = fullfile(cfg.projectRoot,'models','Phase2');
resultsPathP2 = fullfile(cfg.projectRoot,'results','Phase2');
if ~isfolder(resultsPath); mkdir(resultsPath); end
if ~isfolder(figuresPath); mkdir(figuresPath); end

%% 1. Load test data and build evaluation variants
fprintf('Loading test set...\n');
dataPath = P.dataPath;
load(fullfile(dataPath,'wavenumbers.mat'),'wavenumbers_roi');
wavenumbers = wavenumbers_roi;

T = load(fullfile(dataPath,'data_table_test.mat'),'dataTableTest');
dataTableTest = T.dataTableTest;
[X_test, y_test, ~, probeIDs_test] = flatten_spectra_for_pca( ...
    dataTableTest, length(wavenumbers));

testVariants = build_test_variants(X_test, y_test, probeIDs_test, cfg);

%% 2. Locate model sets and Phase 2 results
modelSets = discover_model_sets(modelsPathP2, resultsPathP2);
if isempty(modelSets)
    error('No Phase 2 models found in %s.', modelsPathP2);
end

%% 3. Evaluate models across variants
metricNamesEval = {'Accuracy','Sensitivity_WHO3','Specificity_WHO1','PPV_WHO3','NPV_WHO1','F1_WHO3','F2_WHO3','AUC'};
resultsByVariant = struct('id',{},'description',{},'modelSets',{});
bestModelInfo = struct('variantID',{},'modelSetID',{},'modelName',{},'metrics',{},'modelFile',{});

for v = 1:numel(testVariants)
    variant = testVariants(v);
    fprintf('\nEvaluating test variant: %s\n', variant.description);
    variantResults = struct('modelSetID',{},'modelSetDescription',{},'models',{});
    bestScore = -Inf; bestEntry = struct();

    for s = 1:numel(modelSets)
        modelSet = modelSets(s);
        fprintf('  Model set: %s\n', modelSet.description);
        models = evaluate_model_set(modelSet, variant, wavenumbers, metricNamesEval, figuresPath);
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
    end

    resultsByVariant(v).id = variant.id; %#ok<AGROW>
    resultsByVariant(v).description = variant.description;
    resultsByVariant(v).modelSets = variantResults;
    if ~isempty(fieldnames(bestEntry))
        bestModelInfo(v) = bestEntry; %#ok<AGROW>
    end
end

%% Save combined results
dateStr = string(datetime('now','Format','yyyyMMdd'));
resultsFile = fullfile(resultsPath,sprintf('%s_Phase3_ParallelComparisonResults.mat',dateStr));
save(resultsFile,'resultsByVariant','bestModelInfo','testVariants','modelSets');
fprintf('Saved Phase 3 comparison results to %s\n',resultsFile);

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
            fprintf('Joint T2/Q filtering removed %d/%d test spectra.\n', ...
                outlierStruct.numJointOutliers, numel(keepMask));
        end
    catch ME
        warning('Failed to compute joint outliers on test set: %s', ME.message);
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

function models = evaluate_model_set(modelSet, variant, wavenumbers, metricNamesEval, figuresPath)

    models = struct('name',{},'metrics',{},'modelFile',{},'scores',{},'predicted',{},'probeTable',{},'probeMetrics',{},'CV_Metrics',{},'rocFile',{});
    X = variant.X;
    y = variant.y;
    probeIDs = variant.probeIDs;

    pipeline_names_from_cv = {};
    if ~isempty(modelSet.pipelines)
        pipeline_names_from_cv = cellfun(@(p) p.name, modelSet.pipelines, 'UniformOutput', false);
    end

    for i=1:numel(modelSet.modelFiles)
        mf = fullfile(modelSet.modelFiles(i).folder,modelSet.modelFiles(i).name);
        S = load(mf,'finalModel','aggHyper','selectedIdx','selectedWn','ds'); %#ok<NASGU>
        if ~isfield(S,'finalModel')
            warning('Model file %s missing finalModel. Skipping.', mf);
            continue;
        end
        finalModel = S.finalModel;
        mdlName = finalModel.featureSelectionMethod;
        if isfield(finalModel,'pipelineName'); mdlName = finalModel.pipelineName; end

        [ypred,score] = apply_model_to_data(finalModel,X,wavenumbers);
        posIdx = find(finalModel.LDAModel.ClassNames==3);
        metrics = calculate_performance_metrics(y,ypred,score(:,posIdx),3,metricNamesEval);

        entry = struct();
        entry.name = mdlName;
        entry.metrics = metrics;
        entry.modelFile = mf;
        entry.scores = score(:,posIdx);
        entry.predicted = ypred;

        [probeTable,probeMetrics] = aggregate_probe_metrics(probeIDs,y,score(:,posIdx),ypred,metricNamesEval);
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
        [Xroc,Yroc,~,AUC] = perfcurve(y,score(:,posIdx),3);
        rocFile = fullfile(figuresPath,sprintf('ROC_%s_%s_%s.png', entry.name, modelSet.id, variant.id));
        fig = figure('Visible','off');
        plot(Xroc,Yroc,'LineWidth',1.5); grid on;
        xlabel('False positive rate'); ylabel('True positive rate');
        title(sprintf('ROC - %s (%s, %s) AUC %.3f',entry.name,modelSet.id,variant.id,AUC));
        saveas(fig,rocFile); close(fig);
        entry.rocFile = rocFile;

        models(end+1) = entry; %#ok<AGROW>
    end
end

function [tbl,metrics] = aggregate_probe_metrics(probeIDs,yTrue,scores,yPred,metricNames)
    % probeIDs should be an array of probe identifiers (numeric or string).
    probeIDs = string(probeIDs); % ensure string comparison
    probes = unique(probeIDs,'stable');
    tbl = table();
    tbl.Diss_ID = probes;
    tbl.TrueLabel = zeros(numel(probes),1);
    tbl.MeanProbWHO3 = zeros(numel(probes),1);
    tbl.PredLabel = zeros(numel(probes),1);
    for i=1:numel(probes)
        idx = strcmp(probeIDs,probes(i));
        tbl.TrueLabel(i) = mode(yTrue(idx));
        tbl.MeanProbWHO3(i) = mean(scores(idx));
        tbl.PredLabel(i) = tbl.MeanProbWHO3(i)>0.5; % 0=>WHO1, 1=>WHO3
        tbl.PredLabel(i) = tbl.PredLabel(i).*2+1; % convert 0->1,1->3
    end
    metrics = calculate_performance_metrics(tbl.TrueLabel,tbl.PredLabel,tbl.MeanProbWHO3,3,metricNames);
end
