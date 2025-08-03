function run_phase2_model_selection(cfg)
%RUN_PHASE2_MODEL_SELECTION
%
% Model and feature selection with optional outlier removal.
% Performs nested cross-validation on the chosen training set and
% saves the resulting models and metrics.

fprintf('PHASE 2: Model Selection - %s\n', string(datetime('now')));
if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'projectRoot'); cfg.projectRoot = pwd; end

P = setup_project_paths(cfg.projectRoot,'Phase2');
dataPath = P.dataPath;
resultsPath = P.resultsPath;
modelsPath = P.modelsPath;
if ~isfolder(resultsPath); mkdir(resultsPath); end
if ~isfolder(modelsPath); mkdir(modelsPath); end

%% Load training data
if isfield(cfg,'useOutlierRemoval') && cfg.useOutlierRemoval
    cleanedFiles = dir(fullfile(dataPath,'*_training_set_no_outliers*.mat'));
    if isempty(cleanedFiles)
        error('No cleaned training set found in %s.', dataPath);
    end
    [~,idx] = sort([cleanedFiles.datenum],'descend');
    dataFile = fullfile(cleanedFiles(idx(1)).folder, cleanedFiles(idx(1)).name);
    fprintf('Loading cleaned training data from: %s\n', dataFile);
    tmp = load(dataFile);
    xField = find_field_by_prefix(tmp,'X_train_no_outliers');
    yField = find_field_by_prefix(tmp,'y_train_no_outliers');
    pField = find_field_by_prefix(tmp,'Patient_ID_no_outliers');
    if isempty(xField) || isempty(yField) || isempty(pField)
        error('Expected training variables not found in %s.', dataFile);
    end
    X_full = tmp.(xField);
    y_full = tmp.(yField);
    probeIDs_full = tmp.(pField);
    if isfield(tmp,'wavenumbers_roi')
        wavenumbers_roi = tmp.wavenumbers_roi;
    else
        w_data = load(fullfile(dataPath,'wavenumbers.mat'),'wavenumbers_roi');
        wavenumbers_roi = w_data.wavenumbers_roi;
    end
    if iscolumn(wavenumbers_roi); wavenumbers_roi = wavenumbers_roi'; end
else
    load(fullfile(dataPath,'data_table_train.mat'),'dataTableTrain');
    load(fullfile(dataPath,'wavenumbers.mat'),'wavenumbers_roi');
    if iscolumn(wavenumbers_roi); wavenumbers_roi = wavenumbers_roi'; end
    [X_full, y_full,~, probeIDs_full] = flatten_spectra_for_pca( ...
        dataTableTrain, length(wavenumbers_roi));
end

%% Cross-validation setup
numOuterFolds = 5; numInnerFolds = 3;
[uniqueProbes,~,groupIdx] = unique(probeIDs_full,'stable');
outerCV = cvpartition(uniqueProbes,'KFold',numOuterFolds);
metricNames = {'Accuracy','Sensitivity_WHO3','Specificity_WHO1', ...
    'PPV_WHO3','NPV_WHO1','F1_WHO3','F2_WHO3','AUC'};

%% Define pipelines
pipelines = cell(0,1); pidx=0;

p=struct(); p.name='BaselineLDA'; p.feature_selection_method='none'; p.classifier='LDA';
p.hyperparameters_to_tune={'binningFactor'}; p.binningFactors=[1 2 4 8 16];
pidx=pidx+1; pipelines{pidx}=p;

p=struct(); p.name='FisherLDA'; p.feature_selection_method='fisher'; p.classifier='LDA';
p.hyperparameters_to_tune={'binningFactor','fisherFeaturePercent'};
p.binningFactors=[1 2 4 8 16]; p.fisherFeaturePercent_range=[0.05 0.1 0.2 0.3 0.4 0.5];
pidx=pidx+1; pipelines{pidx}=p;

p=struct(); p.name='PCALDA'; p.feature_selection_method='pca'; p.classifier='LDA';
p.hyperparameters_to_tune={'binningFactor','pcaVarianceToExplain'};
p.binningFactors=[1 2 4 8 16]; p.pcaVarianceToExplain_range=[0.90 0.95 0.99];
pidx=pidx+1; pipelines{pidx}=p;

p=struct(); p.name='MRMRLDA'; p.feature_selection_method='mrmr'; p.classifier='LDA';
p.hyperparameters_to_tune={'binningFactor','mrmrFeaturePercent'};
p.binningFactors=[1 2 4 8 16]; p.mrmrFeaturePercent_range=[0.05 0.1 0.2 0.3 0.4];
pidx=pidx+1; pipelines{pidx}=p;

%% Nested CV
resultsPerPipeline=cell(numel(pipelines),1);
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
        X_tr = X_full(trainMask,:); y_tr = y_full(trainMask); probes_tr = probeIDs_full(trainMask);
        X_te = X_full(testMask,:);  y_te = y_full(testMask);
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
    resultsPerPipeline{iPipe}=res;
    % Train final model on all data
    aggHyper=aggregate_best_hyperparams(outerBestHyper);
    [finalModel,selectedIdx,selectedWn]=train_final_pipeline_model(X_full,y_full,wavenumbers_roi,pipe,aggHyper);
    modelFile=fullfile(modelsPath,sprintf('%s_Phase2_%s_Model.mat',string(datetime('now','Format','yyyyMMdd')),pipe.name));
    save(modelFile,'finalModel','aggHyper','selectedIdx','selectedWn');
    res.finalModelFile=modelFile;
    resultsPerPipeline{iPipe}=res;
end

%% Save results
dateStr=string(datetime('now','Format','yyyyMMdd'));
resultsFile=fullfile(resultsPath,sprintf('%s_Phase2_AllPipelineResults.mat',dateStr));
save(resultsFile,'resultsPerPipeline','pipelines','metricNames','numOuterFolds','numInnerFolds');
fprintf('Phase 2 results saved to %s\n',resultsFile);
end

function [yPred,scores] = apply_model_to_data(model,X,wn)
    Xp = X; currentWn = wn;
    if isfield(model,'binningFactor') && model.binningFactor>1
        [Xp,currentWn] = bin_spectra(X,wn,model.binningFactor);
    end
    switch lower(model.featureSelectionMethod)
        case 'pca'
            Xp = (Xp - model.PCAMu) * model.PCACoeff;
        otherwise
            Xp = Xp(:,model.selectedFeatureIndices);
    end
    [yPred,scores] = predict(model.LDAModel,Xp);
end

function fieldName = find_field_by_prefix(S,prefix)
    fieldName = '';
    fns = fieldnames(S);
    for i=1:numel(fns)
        if startsWith(fns{i}, prefix, 'IgnoreCase', true)
            fieldName = fns{i};
            return;
        end
    end
end
