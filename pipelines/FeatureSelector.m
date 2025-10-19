classdef (Abstract) FeatureSelector
    %FEATURESELECTOR Abstract base class for feature selection components.
    %   Implementations must provide FITTRANSFORM and TRANSFORM methods that
    %   operate on spectral matrices. The base class stores the method name
    %   so downstream logic can introspect the selector type.

    properties (SetAccess = protected)
        Name (1,1) string = "none";
    end

    methods
        function obj = FeatureSelector(name)
            if nargin > 0
                obj.Name = string(name);
            end
        end

        function n = getName(obj)
            n = obj.Name;
        end
    end

    methods (Abstract)
        [obj, Xout, wavenumbersOut] = fitTransform(obj, X, y, wavenumbers, hyperparams)
        [Xout, wavenumbersOut] = transform(obj, X, wavenumbers)
        idx = getSelectedFeatureIndices(obj)
        wn = getSelectedWavenumbers(obj)
    end
end
