classdef FisherFeatureSelector < pipelines.FeatureSelector
    %FISHERFEATURESELECTOR Rank features using Fisher score.

    properties (Access = private)
        SelectedIdx double = [];
        SelectedWavenumbers double = [];
        FeaturePercent (1,1) double = 1.0;
    end

    methods
        function obj = FisherFeatureSelector()
            obj@pipelines.FeatureSelector("fisher");
        end

        function [obj, Xout, wavenumbersOut] = fitTransform(obj, X, y, wavenumbers, hyperparams)
            if nargin < 5 || ~isfield(hyperparams, 'fisherFeaturePercent')
                percent = obj.FeaturePercent;
            else
                percent = hyperparams.fisherFeaturePercent;
            end
            obj.FeaturePercent = percent;

            numFeat = min(size(X,2), max(1, ceil(percent * size(X,2))));
            if numFeat > 0 && size(X,1) > 1 && numel(unique(y)) == 2
                fisherRatios = calculate_fisher_ratio(X, y);
                [~, sortIdx] = sort(fisherRatios, 'descend', 'MissingPlacement', 'last');
                obj.SelectedIdx = sortIdx(1:numFeat);
            else
                obj.SelectedIdx = 1:size(X,2);
            end

            obj.SelectedWavenumbers = wavenumbers(obj.SelectedIdx);
            Xout = X(:, obj.SelectedIdx);
            wavenumbersOut = obj.SelectedWavenumbers;
        end

        function [Xout, wavenumbersOut] = transform(obj, X, wavenumbers)
            if isempty(obj.SelectedIdx)
                Xout = X;
                wavenumbersOut = wavenumbers;
            else
                Xout = X(:, obj.SelectedIdx);
                if isempty(obj.SelectedWavenumbers)
                    wavenumbersOut = wavenumbers(obj.SelectedIdx);
                else
                    wavenumbersOut = obj.SelectedWavenumbers;
                end
            end
        end

        function idx = getSelectedFeatureIndices(obj)
            idx = obj.SelectedIdx;
        end

        function wn = getSelectedWavenumbers(obj)
            wn = obj.SelectedWavenumbers;
        end
    end
end
