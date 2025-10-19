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
            model.selectedFeatureIndices = selectionInfo.selectedFeatureIndices;
            if isempty(selectedWn) && ~isempty(currentWn)
                selectedWn = currentWn(model.selectedFeatureIndices);
            end
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
