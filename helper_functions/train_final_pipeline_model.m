function [model, selectedIdx, selectedWn] = train_final_pipeline_model(X, y, wavenumbers, pipelineConfig, hyperparams)
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
%   Date: 2025-06-16
%
%   This helper consolidates duplicated logic that was previously embedded
%   in the Phase 2 script. It mirrors the preprocessing steps used during
%   cross‑validation so the final model can be applied consistently later on.

    % Default outputs
    model = struct();
    selectedIdx = [];
    selectedWn = [];

    % Ensure wavenumbers row vector
    if iscolumn(wavenumbers)
        wavenumbers = wavenumbers';
    end

    Xp = X; currentWn = wavenumbers;

    % --- Binning ---
    model.binningFactor = 1;
    if isfield(hyperparams, 'binningFactor') && hyperparams.binningFactor > 1
        [Xp, currentWn] = bin_spectra(X, wavenumbers, hyperparams.binningFactor);
        model.binningFactor = hyperparams.binningFactor;
    end

    % --- Feature selection ---
    requestedFeatureMethod = lower(pipelineConfig.feature_selection_method);
    model.requestedFeatureSelectionMethod = requestedFeatureMethod;
    model.featureSelectionMethod = requestedFeatureMethod;
    switch requestedFeatureMethod
        case 'fisher'
            if isfield(hyperparams, 'fisherFeaturePercent')
                numFeat = ceil(hyperparams.fisherFeaturePercent * size(Xp,2));
            else
                numFeat = size(Xp,2);
            end
            numFeat = min(numFeat, size(Xp,2));
            if numFeat > 0 && size(Xp,1) > 1 && length(unique(y)) == 2
                fr = calculate_fisher_ratio(Xp, y);
                [~, sortIdx] = sort(fr, 'descend', 'MissingPlacement','last');
                selectedIdx = sortIdx(1:numFeat);
            else
                selectedIdx = 1:size(Xp,2);
            end
            Xfs = Xp(:, selectedIdx);
            selectedWn = currentWn(selectedIdx);
            model.selectedFeatureIndices = selectedIdx;

        case 'pca'
            Xfs = Xp;
            model.PCACoeff = [];
            model.PCAMu = [];
            if size(Xp,2) > 0 && size(Xp,1) > 1 && size(Xp,1) > size(Xp,2)
                try
                    [coeff, score, ~, ~, explained, mu] = pca(Xp);
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
                catch
                    % fallback to using original features
                    selectedIdx = 1:size(Xp,2);
                    Xfs = Xp;
                end
            else
                selectedIdx = 1:size(Xp,2);
            end
            selectedWn = [];
            model.selectedFeatureIndices = selectedIdx; % not used but keep for completeness
            if isempty(model.PCACoeff) || isempty(model.PCAMu)
                % PCA was not successfully estimated; fall back to using the raw
                % features so downstream application code does not expect the PCA
                % parameters to exist.
                model.featureSelectionMethod = 'none';
            end

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
                catch
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

    % --- Train classifier ---
    switch lower(pipelineConfig.classifier)
        case 'lda'
            mdl = fitcdiscr(Xfs, y);
        otherwise
            error('Unsupported classifier: %s', pipelineConfig.classifier);
    end

    model.LDAModel = mdl;
    model.pipelineName = pipelineConfig.name;
end
