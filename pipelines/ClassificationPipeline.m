classdef ClassificationPipeline
    %CLASSIFICATIONPIPELINE High-level pipeline composed of preprocessing and classifier.

    properties (SetAccess = private)
        Name (1,1) string
        Binner pipelines.BinningTransformer
        FeatureSelector pipelines.FeatureSelector
        Classifier pipelines.LDAClassifier
        HyperparameterOptions struct
        HyperparametersToTune cell
    end

    methods
        function obj = ClassificationPipeline(name, binner, featureSelector, classifier, hyperparameterOptions, hyperparametersToTune)
            arguments
                name (1,1) string
                binner pipelines.BinningTransformer
                featureSelector pipelines.FeatureSelector
                classifier pipelines.LDAClassifier
                hyperparameterOptions struct = struct()
                hyperparametersToTune cell = {}
            end
            obj.Name = name;
            obj.Binner = binner;
            obj.FeatureSelector = featureSelector;
            obj.Classifier = classifier;
            obj.HyperparameterOptions = hyperparameterOptions;
            obj.HyperparametersToTune = hyperparametersToTune;
        end

        function combos = getHyperparameterCombinations(obj)
            defaults = obj.getDefaultHyperparameters();
            names = obj.HyperparametersToTune;
            if isempty(names)
                combos = {defaults};
                return;
            end
            gridCells = cell(1, numel(names));
            for i = 1:numel(names)
                name = names{i};
                if isfield(obj.HyperparameterOptions, name)
                    gridCells{i} = obj.HyperparameterOptions.(name)(:).';
                else
                    error('ClassificationPipeline:MissingHyperparameterOption', ...
                        'No options provided for hyperparameter %s in pipeline %s.', name, obj.Name);
                end
            end
            [gridOutputs{1:numel(gridCells)}] = ndgrid(gridCells{:}); %#ok<CCAT>
            numComb = numel(gridOutputs{1});
            combos = cell(numComb,1);
            for iCombo = 1:numComb
                comboStruct = defaults;
                for j = 1:numel(names)
                    comboStruct.(names{j}) = gridOutputs{j}(iCombo);
                end
                combos{iCombo} = comboStruct;
            end
        end

        function defaults = getDefaultHyperparameters(obj)
            defaults = struct();
            opts = obj.HyperparameterOptions;
            fields = fieldnames(opts);
            for i = 1:numel(fields)
                values = opts.(fields{i});
                if isempty(values)
                    continue;
                end
                defaults.(fields{i}) = values(1);
            end
            if ~isfield(defaults,'binningFactor')
                defaults.binningFactor = 1;
            end
        end

        function merged = mergeHyperparameters(obj, hyperparams)
            defaults = obj.getDefaultHyperparameters();
            merged = defaults;
            if nargin < 2 || isempty(hyperparams)
                return;
            end
            names = fieldnames(hyperparams);
            for i = 1:numel(names)
                merged.(names{i}) = hyperparams.(names{i});
            end
            if ~isfield(merged,'binningFactor')
                merged.binningFactor = defaults.binningFactor;
            end
        end

        function trained = fit(obj, X, y, wavenumbers, hyperparams)
            if nargin < 5
                hyperparams = struct();
            end
            mergedHyper = obj.mergeHyperparameters(hyperparams);

            [binnerTrained, Xb, wnB] = obj.Binner.fitTransform(X, wavenumbers, mergedHyper);
            [featureSelectorTrained, Xfs, wnFs] = obj.FeatureSelector.fitTransform(Xb, y, wnB, mergedHyper);
            classifierTrained = obj.Classifier.fit(Xfs, y, mergedHyper);

            trained = pipelines.TrainedClassificationPipeline(obj.Name, binnerTrained, featureSelectorTrained, classifierTrained, wnFs, mergedHyper);
        end

        function name = getFeatureSelectionMethod(obj)
            name = obj.FeatureSelector.getName();
        end
    end
end
