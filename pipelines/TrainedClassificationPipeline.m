classdef TrainedClassificationPipeline
    %TRAINEDCLASSIFICATIONPIPELINE Stores fitted pipeline components.

    properties (SetAccess = private)
        Name (1,1) string
        Binner pipelines.BinningTransformer
        FeatureSelector pipelines.FeatureSelector
        Classifier pipelines.LDAClassifier
        Hyperparameters struct
        FeatureWavenumbers double = [];
    end

    methods
        function obj = TrainedClassificationPipeline(name, binner, featureSelector, classifier, featureWavenumbers, hyperparams)
            if nargin > 0
                obj.Name = string(name);
                obj.Binner = binner;
                obj.FeatureSelector = featureSelector;
                obj.Classifier = classifier;
                obj.FeatureWavenumbers = featureWavenumbers;
                if nargin >= 6
                    obj.Hyperparameters = hyperparams;
                else
                    obj.Hyperparameters = struct();
                end
            end
        end

        function [labels, scores, classNames] = predict(obj, X, wavenumbers)
            Xt = obj.transform(X, wavenumbers);
            [labels, scores] = obj.Classifier.predict(Xt);
            classNames = obj.Classifier.getClassNames();
        end

        function Xt = transform(obj, X, wavenumbers)
            [Xb, wnB] = obj.Binner.transform(X, wavenumbers);
            [Xt, ~] = obj.FeatureSelector.transform(Xb, wnB);
        end

        function idx = getSelectedFeatureIndices(obj)
            idx = obj.FeatureSelector.getSelectedFeatureIndices();
        end

        function wn = getSelectedWavenumbers(obj)
            wn = obj.FeatureSelector.getSelectedWavenumbers();
            if isempty(wn)
                wn = obj.FeatureWavenumbers;
            end
        end

        function model = getClassifierModel(obj)
            model = obj.Classifier.Model;
        end

        function s = toStruct(obj)
            s = struct();
            s.pipelineName = obj.Name;
            s.hyperparameters = obj.Hyperparameters;
            s.binningFactor = obj.Binner.Factor;
            s.featureSelectionMethod = obj.FeatureSelector.getName();
            s.selectedFeatureIndices = obj.getSelectedFeatureIndices();
            s.selectedWavenumbers = obj.getSelectedWavenumbers();
            if isa(obj.FeatureSelector, 'pipelines.PCAFeatureSelector')
                s.PCACoeff = obj.FeatureSelector.getPCACoeff();
                s.PCAMu = obj.FeatureSelector.getPCAMu();
            end
            s.classifierModel = obj.Classifier.Model;
        end
    end
end
