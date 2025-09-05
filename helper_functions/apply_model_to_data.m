function [yPred,scores] = apply_model_to_data(model,X,wn)
%APPLY_MODEL_TO_DATA Preprocess and apply an LDA model to spectra.
%   [yPred,scores] = APPLY_MODEL_TO_DATA(model,X,wn) bins the spectra if
%   required, performs the configured feature selection transformation and
%   then calls PREDICT on the trained LDA model.
%
% Inputs:
%   model - struct containing fields:
%       binningFactor           - scalar binning factor (optional)
%       featureSelectionMethod  - 'pca' or other supported method
%       selectedFeatureIndices  - indices for non-PCA feature selection
%       PCACoeff, PCAMu         - PCA transformation matrices (when using PCA)
%       LDAModel                - trained classification model with PREDICT
%   X     - matrix of spectra (observations x features)
%   wn    - row vector of wavenumbers corresponding to columns of X
%
% Outputs:
%   yPred - predicted class labels
%   scores - classification scores from the model
%
% This helper consolidates the shared logic used in Phase 2 and Phase 3
% scripts for applying trained models to new data.

    Xp = X; currentWn = wn; %#ok<NASGU>
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
