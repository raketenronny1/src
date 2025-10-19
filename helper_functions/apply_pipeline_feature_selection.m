function X_transformed = apply_pipeline_feature_selection(X, selectionInfo)
%APPLY_PIPELINE_FEATURE_SELECTION Apply fitted feature selection to new data.
%   X_T = APPLY_PIPELINE_FEATURE_SELECTION(X, INFO) transforms the input
%   spectra according to the selectionInfo produced by
%   FIT_PIPELINE_FEATURE_SELECTION.
%
%   Date: 2025-07-07

    if nargin < 2 || isempty(selectionInfo)
        selectionInfo = struct();
    end

    if isempty(fieldnames(selectionInfo))
        X_transformed = X;
        return;
    end

    if ~isfield(selectionInfo, 'method') || isempty(selectionInfo.method)
        method = 'none';
    else
        method = lower(string(selectionInfo.method));
    end
    method = char(method);

    switch method
        case 'pca'
            if ~isfield(selectionInfo, 'PCAMu') || ~isfield(selectionInfo, 'PCACoeff') || isempty(selectionInfo.PCACoeff)
                X_transformed = X;
                return;
            end
            mu = selectionInfo.PCAMu;
            if ~isrow(mu)
                mu = reshape(mu, 1, []);
            end
            if size(mu,2) ~= size(X,2)
                error('apply_pipeline_feature_selection:DimensionMismatch', ...
                      'PCAMu length (%d) does not match input width (%d).', size(mu,2), size(X,2));
            end
            X_transformed = bsxfun(@minus, X, mu) * selectionInfo.PCACoeff;
        otherwise
            if ~isfield(selectionInfo, 'selectedFeatureIndices') || isempty(selectionInfo.selectedFeatureIndices)
                X_transformed = X;
                return;
            end
            idx = selectionInfo.selectedFeatureIndices;
            idx = idx(idx >= 1 & idx <= size(X,2));
            if isempty(idx)
                X_transformed = X;
            else
                X_transformed = X(:, idx);
            end
    end
end
