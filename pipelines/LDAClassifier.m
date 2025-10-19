classdef LDAClassifier
    %LDACLASSIFIER Wrapper around fitcdiscr to provide a consistent interface.

    properties (SetAccess = private)
        Model = [];
    end

    methods
        function obj = LDAClassifier(model)
            if nargin > 0
                obj.Model = model;
            end
        end

        function obj = fit(obj, X, y, ~)
            obj.Model = fitcdiscr(X, y);
        end

        function [labels, scores] = predict(obj, X)
            [labels, scores] = predict(obj.Model, X);
        end

        function classNames = getClassNames(obj)
            if isempty(obj.Model)
                classNames = [];
            else
                classNames = obj.Model.ClassNames;
            end
        end
    end
end
