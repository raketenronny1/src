function [yPred,scores] = apply_model_to_data(model,X,wn)
%APPLY_MODEL_TO_DATA Preprocess and apply an LDA model to spectra.
%   [yPred,scores] = APPLY_MODEL_TO_DATA(model,X,wn) bins the spectra if
%   required, performs the configured feature selection transformation and
%   then calls PREDICT on the trained LDA model.
%
% Inputs:
%   model - struct containing fields:
%       binningFactor           - scalar binning factor (optional)
%       binningChunkSize        - optional chunk size to reuse during binning
%       featureSelectionMethod  - 'pca' or other supported method
%       selectedFeatureIndices  - indices for non-PCA feature selection
%       PCACoeff, PCAMu         - PCA transformation matrices (when using PCA)
%       LDAModel                - trained classification model with PREDICT
%   X     - matrix of spectra (observations x features)
%   wn    - row vector of wavenumbers corresponding to columns of X
%
% Outputs:
%   yPred  - predicted class labels
%   scores - classification scores from the model
%
% This helper consolidates the shared logic used in Phase 2 and Phase 3
% scripts for applying trained models to new data.

    Xp = X;
    binChunkSize = [];
    if isfield(model,'binningChunkSize')
        binChunkSize = model.binningChunkSize;
    end

    if isfield(model,'binningFactor') && model.binningFactor>1
        Xp = bin_spectra(X,wn,model.binningFactor,'ChunkSize',binChunkSize);
    end
    featureMethod = 'none';
    if isfield(model,'featureSelectionMethod') && ~isempty(model.featureSelectionMethod)
        featureMethod = lower(model.featureSelectionMethod);
    end

    switch featureMethod
        case 'pca'
            mu = model.PCAMu;
            if ~isrow(mu)
                mu = reshape(mu, 1, []);
            end
            if size(mu,2) ~= size(Xp,2)
                error('apply_model_to_data:DimensionMismatch', ...
                      'PCAMu length (%d) does not match data width (%d).', ...
                      size(mu,2), size(Xp,2));
            end
            Xp = bsxfun(@minus, Xp, mu) * model.PCACoeff;
        case {'none',''}
            % no additional feature processing
        otherwise
            if ~isfield(model,'selectedFeatureIndices') || isempty(model.selectedFeatureIndices)
                error('apply_model_to_data:MissingFeatureIndices', ...
                      'Model is missing selectedFeatureIndices for method %s.', featureMethod);
            end
            Xp = Xp(:,model.selectedFeatureIndices);
    end
    [yPred,scores] = predict(model.LDAModel,Xp);
end
