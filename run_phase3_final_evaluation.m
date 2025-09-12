function run_phase3_final_evaluation(cfg)
%RUN_PHASE3_FINAL_EVALUATION
%
% Apply each pipeline model saved in Phase 2 to the test data and compare
% performance. Outlier removal is no longer performed so the script simply
% loads the available Phase 2 models, evaluates them on the unseen test set
% and compares the results with cross‑validation metrics.
%
% Date: 2025-06-10

%% 0. Configuration
if nargin < 1
    cfg = struct();
end
if ~isfield(cfg,'projectRoot'); cfg.projectRoot = pwd; end

P = setup_project_paths(cfg.projectRoot,'Phase3');
resultsPath = P.resultsPath;
figuresPath = P.figuresPath;
modelsPathP2 = fullfile(cfg.projectRoot,'models','Phase2');
resultsPathP2 = fullfile(cfg.projectRoot,'results','Phase2');
if ~isfolder(resultsPath); mkdir(resultsPath); end
if ~isfolder(figuresPath); mkdir(figuresPath); end

%% 1. Load test data
fprintf('Loading test set...\n');
dataPath = P.dataPath;
load(fullfile(dataPath,'wavenumbers.mat'),'wavenumbers_roi');
wavenumbers = wavenumbers_roi;

T = load(fullfile(dataPath,'data_table_test.mat'),'dataTableTest');
dataTableTest = T.dataTableTest;
numProbes = height(dataTableTest);
X_list = cell(numProbes,1);
y_list = cell(numProbes,1);
probe_list = cell(numProbes,1);
for i=1:numProbes
    X_list{i} = dataTableTest.CombinedSpectra{i};
    lbl = dataTableTest.WHO_Grade(i);
    if lbl=="WHO-1"; y_tmp = 1; else; y_tmp = 3; end
    y_list{i} = repmat(y_tmp,size(X_list{i},1),1);
    probe_list{i} = repmat(dataTableTest.Diss_ID(i),size(X_list{i},1),1);
end
X_test = vertcat(X_list{:});
y_test = vertcat(y_list{:});
probeIDs_test = vertcat(probe_list{:});

%% 2. Locate models and Phase2 results
modelFiles = dir(fullfile(modelsPathP2, '*_Phase2_*_Model.mat'));
if isempty(modelFiles)
    error('No Phase 2 models found in %s.', modelsPathP2);
end
% Keep only the most recent model file for each pipeline to avoid duplicates
fileMap = containers.Map();
for i = 1:numel(modelFiles)
    tokens = regexp(modelFiles(i).name,'^\d+_Phase2_(.+)_Model\.mat$','tokens','once');
    if isempty(tokens); continue; end
    pipe = tokens{1};
    if ~isKey(fileMap, pipe) || modelFiles(i).datenum > fileMap(pipe).datenum
        fileMap(pipe) = modelFiles(i);
    end
end
modelFilesCell = values(fileMap);
modelFiles = [modelFilesCell{:}];
[~,order] = sort({modelFiles.name});
modelFiles = modelFiles(order);
resFile = dir(fullfile(resultsPathP2,'*_Phase2_AllPipelineResults.mat'));
if isempty(resFile)
    warning('Phase 2 results file not found.');
    cvData = [];
else
    [~,idx] = sort([resFile.datenum],'descend');
    tmp = load(fullfile(resFile(idx(1)).folder,resFile(idx(1)).name));
    cvData = tmp.resultsPerPipeline;
    pipelinesCV = tmp.pipelines;
    metricNames = tmp.metricNames;
end

%% 3. Evaluate each model
metricNamesEval = {'Accuracy','Sensitivity_WHO3','Specificity_WHO1','PPV_WHO3','NPV_WHO1','F1_WHO3','F2_WHO3','AUC'};
results = struct();
for i=1:numel(modelFiles)
    mf = fullfile(modelFiles(i).folder,modelFiles(i).name);
    load(mf,'finalModel','aggHyper','selectedIdx','selectedWn');
    mdlName = finalModel.featureSelectionMethod;
    if isfield(finalModel,'pipelineName'); mdlName = finalModel.pipelineName; end

    [ypred,score] = apply_model_to_data(finalModel,X_test,wavenumbers);
    posIdx = find(finalModel.LDAModel.ClassNames==3);
    m = calculate_performance_metrics(y_test,ypred,score(:,posIdx),3,metricNamesEval);
    results(i).name = mdlName; %#ok<*AGROW>
    results(i).metrics = m;
    results(i).modelFile = mf;
    results(i).scores = score(:,posIdx);
    results(i).predicted = ypred;

    % probe level
    [probeTable,probeMetrics] = aggregate_probe_metrics(probeIDs_test,y_test,score(:,posIdx),ypred,metricNamesEval);
    results(i).probeTable = probeTable;
    results(i).probeMetrics = probeMetrics;

    % ROC curve
    [Xroc,Yroc,~,AUC] = perfcurve(y_test,score(:,posIdx),3);
    fig = figure('Visible','off');
    plot(Xroc,Yroc,'LineWidth',1.5); grid on;
    xlabel('False positive rate'); ylabel('True positive rate');
    title(sprintf('ROC - %s (AUC %.3f)',mdlName,AUC));
    rocFile = fullfile(figuresPath,sprintf('ROC_%s.png',mdlName));
    saveas(fig,rocFile); close(fig);
    results(i).rocFile = rocFile;
end

%% 4. Compare with CV metrics if available
if ~isempty(cvData)
    % Extract the pipeline names from the cell array of structs into a new cell array.
    % This is the corrected part.
    pipeline_names_from_cv = cellfun(@(p) p.name, pipelinesCV, 'UniformOutput', false);

    for i=1:numel(results)
        % Now, compare the name of the current test result with the list of names from CV.
        idx = find(strcmpi(pipeline_names_from_cv, results(i).name));
        if ~isempty(idx)
            results(i).CV_Metrics = cvData{idx}.outerFoldMetrics_mean;
        end
    end
end

%% Determine best model based on F2_WHO3 on test set
bestScore = -Inf; bestIdx = 1;
for i=1:numel(results)
    if results(i).metrics.F2_WHO3 > bestScore
        bestScore = results(i).metrics.F2_WHO3;
        bestIdx = i;
    end
end
bestModelInfo = results(bestIdx);

%% Save combined results
dateStr = string(datetime('now','Format','yyyyMMdd'));
resultsFile = fullfile(resultsPath,sprintf('%s_Phase3_ComparisonResults.mat',dateStr));
save(resultsFile,'results','bestModelInfo');
fprintf('Saved Phase 3 comparison results to %s\n',resultsFile);

end

%% Helper functions
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
