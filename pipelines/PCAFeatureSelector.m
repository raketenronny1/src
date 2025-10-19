classdef PCAFeatureSelector < pipelines.FeatureSelector
    %PCAFEATURESELECTOR Dimensionality reduction via PCA.

    properties (Access = private)
        Coeff double = [];
        Mu double = [];
        NumComponents (1,1) double = 0;
    end

    methods
        function obj = PCAFeatureSelector()
            obj@pipelines.FeatureSelector("pca");
        end

        function [obj, Xout, wavenumbersOut] = fitTransform(obj, X, ~, wavenumbers, hyperparams)
            Xout = X;
            wavenumbersOut = [];
            obj.Coeff = [];
            obj.Mu = [];
            obj.NumComponents = 0;

            if isempty(X) || size(X,1) <= 1 || size(X,1) <= size(X,2)
                return;
            end

            [coeff, score, ~, ~, explained, mu] = pca(X);
            nComp = size(coeff,2);
            if nargin >= 5 && isfield(hyperparams,'pcaVarianceToExplain')
                cumExp = cumsum(explained);
                idx = find(cumExp >= hyperparams.pcaVarianceToExplain * 100, 1, 'first');
                if ~isempty(idx)
                    nComp = idx;
                end
            elseif nargin >=5 && isfield(hyperparams,'numPCAComponents')
                nComp = min(hyperparams.numPCAComponents, size(coeff,2));
            end
            coeff = coeff(:,1:nComp);
            score = score(:,1:nComp);

            obj.Coeff = coeff;
            obj.Mu = mu;
            obj.NumComponents = nComp;
            Xout = score;
        end

        function [Xout, wavenumbersOut] = transform(obj, X, ~)
            if isempty(obj.Coeff)
                Xout = X;
                wavenumbersOut = [];
                return;
            end
            mu = obj.Mu;
            if ~isrow(mu)
                mu = reshape(mu, 1, []);
            end
            Xout = bsxfun(@minus, X, mu) * obj.Coeff;
            wavenumbersOut = [];
        end

        function idx = getSelectedFeatureIndices(obj)
            if obj.NumComponents == 0
                idx = [];
            else
                idx = 1:obj.NumComponents;
            end
        end

        function wn = getSelectedWavenumbers(obj)
            wn = [];
        end

        function coeff = getPCACoeff(obj)
            coeff = obj.Coeff;
        end

        function mu = getPCAMu(obj)
            mu = obj.Mu;
        end
    end
end
