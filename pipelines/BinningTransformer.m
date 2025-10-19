classdef BinningTransformer
    %BINNINGTRANSFORMER Optional spectral binning component.

    properties (SetAccess = private)
        Factor (1,1) double = 1;
    end

    methods
        function obj = BinningTransformer(defaultFactor)
            if nargin > 0
                obj.Factor = defaultFactor;
            end
        end

        function [obj, Xout, wavenumbersOut] = fitTransform(obj, X, wavenumbers, hyperparams)
            factor = obj.Factor;
            if nargin >= 4 && isfield(hyperparams, 'binningFactor')
                factor = hyperparams.binningFactor;
            end
            obj.Factor = factor;
            [Xout, wavenumbersOut] = obj.transform(X, wavenumbers);
        end

        function [Xout, wavenumbersOut] = transform(obj, X, wavenumbers)
            if obj.Factor > 1
                [Xout, wavenumbersOut] = bin_spectra(X, wavenumbers, obj.Factor);
            else
                Xout = X;
                wavenumbersOut = wavenumbers;
            end
        end
    end
end
