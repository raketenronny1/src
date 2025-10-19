classdef NoFeatureSelector < pipelines.FeatureSelector
    %NOFEATURESELECTOR Pass-through feature selector.

    properties (Access = private)
        NumFeatures (1,1) double = 0;
        CachedWavenumbers double = [];
    end

    methods
        function obj = NoFeatureSelector()
            obj@pipelines.FeatureSelector("none");
        end

        function [obj, Xout, wavenumbersOut] = fitTransform(obj, X, ~, wavenumbers, ~)
            obj.NumFeatures = size(X, 2);
            obj.CachedWavenumbers = wavenumbers;
            Xout = X;
            wavenumbersOut = wavenumbers;
        end

        function [Xout, wavenumbersOut] = transform(obj, X, wavenumbers)
            Xout = X;
            if isempty(obj.CachedWavenumbers)
                wavenumbersOut = wavenumbers;
            else
                wavenumbersOut = obj.CachedWavenumbers;
            end
        end

        function idx = getSelectedFeatureIndices(obj)
            if obj.NumFeatures == 0
                idx = [];
            else
                idx = 1:obj.NumFeatures;
            end
        end

        function wn = getSelectedWavenumbers(obj)
            wn = obj.CachedWavenumbers;
        end
    end
end
