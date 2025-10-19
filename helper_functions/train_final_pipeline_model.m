function [model, selectedIdx, selectedWn, diagnostics] = train_final_pipeline_model(X, y, wavenumbers, pipelineConfig, hyperparams)
%TRAIN_FINAL_PIPELINE_MODEL Train a final model for a given pipeline.
%   [MODEL, IDX, WN] = TRAIN_FINAL_PIPELINE_MODEL(X, Y, WN, PIPE, HYPER)
%   bins the spectra if required, performs feature selection according to the
%   pipeline specification and trains the configured classifier. The helper
%   returns the model struct along with the indices of selected features and
%   their corresponding wavenumbers.
%
%   Inputs
%       X              - training spectra (observations x features)
%       y              - class labels
%       wavenumbers    - row vector of wavenumbers
%       pipelineConfig - struct describing the pipeline
%       hyperparams    - struct of chosen hyperparameter values
%
%   Outputs
%       model      - struct containing fields used by APPLY_MODEL_TO_DATA
%       selectedIdx- indices of selected features after preprocessing
%       selectedWn - wavenumbers corresponding to selectedIdx
%
%   Optional inputs
%       chunkOptions.binSpectraRows - rows per chunk during binning
%       chunkOptions.fisherPerClass - rows per chunk per class for Fisher ratios
%
%   Date: 2025-06-16
%
%   This helper consolidates duplicated logic that was previously embedded
%   in the Phase 2 script. It mirrors the preprocessing steps used during
%   cross‑validation so the final model can be applied consistently later on.

    if iscolumn(wavenumbers)
        wavenumbers = wavenumbers';
    end

    if isa(pipelineConfig, 'pipelines.ClassificationPipeline')
        if nargin < 5
            hyperparams = struct();
        end
        trainedPipeline = pipelineConfig.fit(X, y, wavenumbers, hyperparams);
        model = trainedPipeline;
        selectedIdx = trainedPipeline.getSelectedFeatureIndices();
        selectedWn = trainedPipeline.getSelectedWavenumbers();
        return;
    end

    % Default outputs
    model = struct();
    selectedIdx = [];
    selectedWn = [];
    diagnostics = init_diagnostics();
    baseContext = sprintf('train_final_pipeline_model:%s', pipelineConfig.name);

    % Ensure wavenumbers row vector
    if iscolumn(wavenumbers)
        wavenumbers = wavenumbers';
    end

    [Xp, currentWn, preprocessInfo] = apply_pipeline_preprocessing(X, wavenumbers, hyperparams);
    model.binningFactor = preprocessInfo.binningFactor;

    [Xfs, selectionInfo] = fit_pipeline_feature_selection(Xp, y, pipelineConfig, hyperparams, currentWn);
    model.featureSelectionMethod = pipelineConfig.feature_selection_method;
    selectedIdx = selectionInfo.selectedFeatureIndices;
    selectedWn = selectionInfo.selectedWavenumbers;

    switch lower(model.featureSelectionMethod)
        case 'pca'
            model.PCACoeff = selectionInfo.PCACoeff;
            model.PCAMu = selectionInfo.PCAMu;
            model.selectedFeatureIndices = selectionInfo.selectedFeatureIndices;
        otherwise
            model.PCACoeff = [];
            model.PCAMu = [];
            if size(Xp,2) > 0 && size(Xp,1) > 1 && size(Xp,1) > size(Xp,2)
                try
                    cacheConfig = struct('signature', struct( ...
                        'context', 'train_final_pipeline_model', ...
                        'hyperparams', hyperparams));
                    [coeff, score, ~, ~, explained, mu] = cached_pca(Xp, cacheConfig);
                    if isfield(hyperparams, 'pcaVarianceToExplain')
                        cumExp = cumsum(explained);
                        nComp = find(cumExp >= hyperparams.pcaVarianceToExplain*100, 1, 'first');
                        if isempty(nComp); nComp = size(coeff,2); end
                    elseif isfield(hyperparams,'numPCAComponents')
                        nComp = min(hyperparams.numPCAComponents, size(coeff,2));
                    else
                        nComp = size(coeff,2);
                    end
                    coeff = coeff(:,1:nComp); score = score(:,1:nComp);
                    model.PCACoeff = coeff;
                    model.PCAMu = mu;
                    Xfs = score;
                    selectedIdx = 1:nComp;
                catch ME_pca
                    entry = log_pipeline_message('warning', sprintf('%s:PCA', baseContext), ...
                        'PCA failed (%s). Reverting to original features.', ME_pca.message);
                    diagnostics = record_diagnostic(diagnostics, entry, ME_pca, 'warning');
                    selectedIdx = 1:size(Xp,2);
                    Xfs = Xp;
                end
            else
                selectedIdx = 1:size(Xp,2);
            end
            selectedWn = [];
            model.selectedFeatureIndices = selectedIdx; % not used but keep for completeness

        case 'mrmr'
            if isfield(hyperparams, 'mrmrFeaturePercent')
                numFeat = ceil(hyperparams.mrmrFeaturePercent * size(Xp,2));
            else
                numFeat = size(Xp,2);
            end
            numFeat = min(numFeat, size(Xp,2));
            if numFeat>0 && size(Xp,1)>1 && length(unique(y))==2 && exist('fscmrmr','file')
                try
                    ycat = categorical(y);
                    [idx,~] = fscmrmr(Xp, ycat);
                    numFeat = min(numFeat, length(idx));
                    selectedIdx = idx(1:numFeat);
                catch ME_mrmr
                    entry = log_pipeline_message('warning', sprintf('%s:MRMR', baseContext), ...
                        'fscmrmr failed (%s). Using all features.', ME_mrmr.message);
                    diagnostics = record_diagnostic(diagnostics, entry, ME_mrmr, 'warning');
                    selectedIdx = 1:size(Xp,2);
                end
            else
                selectedIdx = 1:size(Xp,2);
            end
            Xfs = Xp(:,selectedIdx);
            selectedWn = currentWn(selectedIdx);
            model.selectedFeatureIndices = selectedIdx;

        otherwise
            selectedIdx = 1:size(Xp,2);
            Xfs = Xp;
            selectedWn = currentWn(selectedIdx);
            model.selectedFeatureIndices = selectedIdx;
    end
    model.selectedWavenumbers = selectedWn;

    % --- Train classifier ---
    switch lower(pipelineConfig.classifier)
        case 'lda'
            try
                mdl = fitcdiscr(Xfs, y);
            catch ME_lda
                entry = log_pipeline_message('error', sprintf('%s:LDA', baseContext), ...
                    'Failed to train LDA classifier: %s', ME_lda.message);
                diagnostics = record_diagnostic(diagnostics, entry, ME_lda, 'error');
                model = struct();
                selectedIdx = [];
                selectedWn = [];
                return;
            end
        otherwise
            msg = sprintf('Unsupported classifier: %s', pipelineConfig.classifier);
            entry = log_pipeline_message('error', sprintf('%s:Classifier', baseContext), msg);
            diagnostics = record_diagnostic(diagnostics, entry, [], 'error');
            error(msg);
    end

    model.LDAModel = mdl;
    model.pipelineName = pipelineConfig.name;
    if strcmpi(diagnostics.status, 'ok')
        log_pipeline_message('info', baseContext, 'Successfully trained pipeline.');
    end
end

function diagnostics = init_diagnostics()
    diagnostics = struct();
    diagnostics.status = 'ok';
    diagnostics.entries = struct('timestamp',{},'level',{},'context',{},'message',{});
    diagnostics.errors = {};
end

function diagnostics = record_diagnostic(diagnostics, entry, exceptionObj, level)
    diagnostics.entries(end+1) = entry; %#ok<AGROW>
    if nargin >= 3 && ~isempty(exceptionObj)
        diagnostics.errors{end+1} = exceptionObj; %#ok<AGROW>
    end
    switch lower(level)
        case 'error'
            diagnostics.status = 'error';
        case 'warning'
            if ~strcmpi(diagnostics.status, 'error')
                diagnostics.status = 'warning';
            end
    end
end
