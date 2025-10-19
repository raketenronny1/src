function [X_selected, selectionInfo] = fit_pipeline_feature_selection(X, y, pipelineConfig, hyperparams, wavenumbers)
%FIT_PIPELINE_FEATURE_SELECTION Fit feature selection for a pipeline.
%   [X_SELECTED, INFO] = FIT_PIPELINE_FEATURE_SELECTION(X, Y, PIPECFG, HYPER, WN)
%   performs the configured feature selection (Fisher, PCA, MRMR, or none)
%   and returns both the transformed training data and a struct describing
%   how to apply the transformation to new data.
%
%   The returned INFO struct contains:
%       .method                  - lowercase feature selection method
%       .selectedFeatureIndices  - indices used for non-PCA methods
%       .selectedWavenumbers     - wavenumbers corresponding to selected indices
%       .PCACoeff, .PCAMu        - PCA parameters when method == 'pca'
%       .numComponents           - number of PCA components retained
%
%   Date: 2025-07-07

    if nargin < 4 || isempty(hyperparams)
        hyperparams = struct();
    end
    if nargin < 5
        wavenumbers = [];
    end
    if iscolumn(wavenumbers)
        wavenumbers = wavenumbers';
    end

    selectionInfo = struct();
    if ~isfield(pipelineConfig, 'feature_selection_method') || isempty(pipelineConfig.feature_selection_method)
        method = 'none';
    else
        method = lower(string(pipelineConfig.feature_selection_method));
    end
    method = char(method);
    selectionInfo.method = method;

    numFeatures = size(X, 2);
    selectionInfo.selectedFeatureIndices = 1:numFeatures;
    selectionInfo.selectedWavenumbers = [];
    selectionInfo.PCACoeff = [];
    selectionInfo.PCAMu = [];
    selectionInfo.numComponents = [];

    switch method
        case 'fisher'
            if numFeatures == 0
                X_selected = X;
                selectionInfo.selectedFeatureIndices = [];
                selectionInfo.selectedWavenumbers = [];
                return;
            end
            if isfield(hyperparams, 'fisherFeaturePercent')
                numFeat = ceil(hyperparams.fisherFeaturePercent * numFeatures);
            else
                numFeat = numFeatures;
            end
            numFeat = max(0, min(numFeat, numFeatures));

            if numFeat > 0 && size(X,1) > 1 && length(unique(y)) == 2
                fisherRatios = calculate_fisher_ratio(X, y);
                [~, sortedIdx] = sort(fisherRatios, 'descend', 'MissingPlacement', 'last');
                actualNum = min(numFeat, numel(sortedIdx));
                selectionInfo.selectedFeatureIndices = sortedIdx(1:actualNum);
            else
                selectionInfo.selectedFeatureIndices = 1:numFeatures;
            end
            X_selected = X(:, selectionInfo.selectedFeatureIndices);
            if ~isempty(wavenumbers)
                selectionInfo.selectedWavenumbers = wavenumbers(selectionInfo.selectedFeatureIndices);
            end

        case 'pca'
            selectionInfo.selectedWavenumbers = [];
            X_selected = X;
            if numFeatures == 0
                selectionInfo.selectedFeatureIndices = [];
                return;
            end
            if size(X,1) > 1 && size(X,1) > numFeatures
                try
                    [coeff, score, ~, ~, explained, mu] = pca(X);
                    if isfield(hyperparams, 'pcaVarianceToExplain')
                        cumulativeExplained = cumsum(explained);
                        idx_pc = find(cumulativeExplained >= hyperparams.pcaVarianceToExplain * 100, 1, 'first');
                        if isempty(idx_pc)
                            idx_pc = size(coeff, 2);
                        end
                    elseif isfield(hyperparams, 'numPCAComponents')
                        idx_pc = min(hyperparams.numPCAComponents, size(coeff,2));
                    else
                        idx_pc = size(coeff, 2);
                    end
                    if idx_pc < 1
                        idx_pc = 1;
                    end
                    selectionInfo.PCACoeff = coeff(:,1:idx_pc);
                    selectionInfo.PCAMu = mu;
                    selectionInfo.numComponents = idx_pc;
                    selectionInfo.selectedFeatureIndices = 1:idx_pc;
                    X_selected = score(:,1:idx_pc);
                catch
                    % Fall through to use original data
                    selectionInfo.PCACoeff = [];
                    selectionInfo.PCAMu = [];
                    selectionInfo.numComponents = [];
                    selectionInfo.selectedFeatureIndices = 1:numFeatures;
                    X_selected = X;
                end
            else
                selectionInfo.selectedFeatureIndices = 1:numFeatures;
            end

        case 'mrmr'
            if numFeatures == 0
                X_selected = X;
                selectionInfo.selectedFeatureIndices = [];
                selectionInfo.selectedWavenumbers = [];
                return;
            end
            if isfield(hyperparams, 'mrmrFeaturePercent')
                numFeat = ceil(hyperparams.mrmrFeaturePercent * numFeatures);
            else
                numFeat = numFeatures;
            end
            numFeat = max(0, min(numFeat, numFeatures));
            if numFeat <= 0 || size(X,1) <= 1 || length(unique(y)) ~= 2 || ~exist('fscmrmr','file')
                selectionInfo.selectedFeatureIndices = 1:numFeatures;
            else
                try
                    ycat = categorical(y);
                    rankedIdx = fscmrmr(X, ycat);
                    actualNum = min(numFeat, numel(rankedIdx));
                    if actualNum > 0
                        selectionInfo.selectedFeatureIndices = rankedIdx(1:actualNum);
                    else
                        selectionInfo.selectedFeatureIndices = 1:numFeatures;
                    end
                catch
                    selectionInfo.selectedFeatureIndices = 1:numFeatures;
                end
            end
            X_selected = X(:, selectionInfo.selectedFeatureIndices);
            if ~isempty(wavenumbers)
                selectionInfo.selectedWavenumbers = wavenumbers(selectionInfo.selectedFeatureIndices);
            end

        otherwise
            % 'none' or unrecognised method -> pass-through
            selectionInfo.selectedFeatureIndices = 1:numFeatures;
            X_selected = X;
            if ~isempty(wavenumbers) && numFeatures > 0
                selectionInfo.selectedWavenumbers = wavenumbers(selectionInfo.selectedFeatureIndices);
            end
    end
end
