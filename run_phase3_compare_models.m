function summary = run_phase3_compare_models(cfg)
%RUN_PHASE3_COMPARE_MODELS Compare two Phase 3 model outputs via paired t-test.
%   SUMMARY = RUN_PHASE3_COMPARE_MODELS(CFG) locates a Phase 3
%   ParallelComparisonResults MAT-file, extracts two model result structs and
%   runs COMPARE_RESULTS to evaluate whether their WHO-3 probabilities differ
%   significantly. The helper prints a textual summary (unless cfg.quiet is
%   true) and returns the structured output from COMPARE_RESULTS.
%
%   Requires the Statistics and Machine Learning Toolbox (for TTEST).
%
%   CFG fields (all optional):
%       projectRoot         - Repository root (default: get_project_root()).
%       phase3ResultsFile   - Explicit path to a Phase 3 results MAT-file.
%       variantID           - ID or description of the test variant to use.
%       modelSetID          - ID or description of the model set to use.
%       modelA              - Name (string) or index (numeric) of the first model.
%       modelB              - Name (string) or index (numeric) of the second model.
%       mode                - 'spectra' or 'probe' (default 'probe').
%       alpha               - Significance level for the t-test (default 0.05).
%       quiet               - true to suppress textual output.
%
%   Example:
%       cfg = struct('modelA', 'MRMR_LDA', 'modelB', 'MRMR_LDA_Cleaned');
%       run_phase3_compare_models(cfg);
%
%   See also: RUN_PHASE3_FINAL_EVALUATION, COMPARE_RESULTS
arguments
    cfg.projectRoot string = string(get_project_root())
    cfg.phase3ResultsFile string = ""
    cfg.variantID string = ""
    cfg.modelSetID string = ""
    cfg.modelA = []
    cfg.modelB = []
    cfg.mode (1,1) string {mustBeMember(cfg.mode,["spectra","probe"])} = "probe"
    cfg.alpha (1,1) double {mustBePositive} = 0.05
    cfg.quiet (1,1) logical = false
end

projectRoot = char(cfg.projectRoot);
setup_project_paths(projectRoot);

resultsFile = resolve_results_file(projectRoot, cfg.phase3ResultsFile);

S = load(resultsFile, 'resultsByVariant');
if ~isfield(S, 'resultsByVariant') || isempty(S.resultsByVariant)
    error('run_phase3_compare_models:MissingData', ...
        'The file %s does not contain resultsByVariant.', resultsFile);
end

[variant, variantLabel] = select_variant(S.resultsByVariant, cfg.variantID);
[modelSet, modelSetLabel] = select_model_set(variant.modelSets, cfg.modelSetID);

[modelA, labelA] = select_model(modelSet.models, cfg.modelA);
[modelB, labelB] = select_model(modelSet.models, cfg.modelB, 'excludeIndex', modelA.index);

if ~cfg.quiet
    fprintf('Comparing models %s and %s within variant "%s" (%s).\n', ...
        labelA, labelB, variantLabel, modelSetLabel);
end

summary = compare_results(modelA.entry, modelB.entry, struct( ...
    'mode', cfg.mode, ...
    'labels', [labelA, labelB], ...
    'alpha', cfg.alpha));

if ~cfg.quiet
    display_summary(summary);
end

end

function resultsFile = resolve_results_file(projectRoot, overridePath)
    if strlength(overridePath) > 0
        resultsFile = char(overridePath);
        if ~isfile(resultsFile)
            error('run_phase3_compare_models:MissingFile', 'Specified file %s does not exist.', resultsFile);
        end
        return;
    end

    resultsDir = fullfile(projectRoot, 'results', 'Phase3');
    if ~isfolder(resultsDir)
        error('run_phase3_compare_models:MissingDirectory', ...
            'Results directory %s does not exist.', resultsDir);
    end

    files = dir(fullfile(resultsDir, '*_Phase3_ParallelComparisonResults.mat'));
    if isempty(files)
        error('run_phase3_compare_models:NoResults', ...
            'No Phase 3 comparison files found in %s.', resultsDir);
    end

    [~, idx] = max([files.datenum]);
    resultsFile = fullfile(files(idx).folder, files(idx).name);
end

function [variant, label] = select_variant(variants, identifier)
    if isempty(variants)
        error('run_phase3_compare_models:NoVariants', 'No variants available in the results file.');
    end

    if strlength(identifier) == 0
        variant = variants(1);
        label = string(getfield_or(variant, 'description', getfield_or(variant, 'id', 'Variant1')));
        return;
    end

    identifierLower = lower(char(identifier));
    for i = 1:numel(variants)
        idValue = string(getfield_or(variants(i), 'id', ''));
        descValue = string(getfield_or(variants(i), 'description', ''));
        if strcmpi(idValue, string(identifierLower)) || contains(lower(descValue), identifierLower)
            variant = variants(i);
            label = descValue;
            if strlength(label) == 0
                label = idValue;
            end
            return;
        end
    end

    error('run_phase3_compare_models:VariantNotFound', ...
        'Variant "%s" not found in the results file.', identifier);
end

function [modelSet, label] = select_model_set(modelSets, identifier)
    if isempty(modelSets)
        error('run_phase3_compare_models:NoModelSets', ...
            'No model sets present for the selected variant.');
    end

    if strlength(identifier) == 0
        modelSet = modelSets(1);
        label = string(getfield_or(modelSet, 'modelSetDescription', getfield_or(modelSet, 'modelSetID', 'ModelSet1')));
        return;
    end

    identifierLower = lower(char(identifier));
    for i = 1:numel(modelSets)
        idValue = string(getfield_or(modelSets(i), 'modelSetID', ''));
        descValue = string(getfield_or(modelSets(i), 'modelSetDescription', ''));
        if strcmpi(idValue, string(identifierLower)) || contains(lower(descValue), identifierLower)
            modelSet = modelSets(i);
            label = descValue;
            if strlength(label) == 0
                label = idValue;
            end
            return;
        end
    end

    error('run_phase3_compare_models:ModelSetNotFound', ...
        'Model set "%s" not found for the selected variant.', identifier);
end

function [modelInfo, label] = select_model(models, selector, opts)
arguments
    models (1,:) struct
    selector = []
    opts.excludeIndex (1,1) double = NaN
end

    if isempty(models)
        error('run_phase3_compare_models:NoModels', 'No models available in the selected model set.');
    end

    indices = 1:numel(models);
    if ~isnan(opts.excludeIndex)
        indices(indices == opts.excludeIndex) = [];
        if isempty(indices)
            error('run_phase3_compare_models:InsufficientModels', ...
                'Need at least two distinct models to compare.');
        end
    end

    chosenIdx = [];
    names = string({models.name});

    if isempty(selector)
        chosenIdx = indices(1);
    elseif isnumeric(selector)
        chosenIdx = selector;
    elseif isstring(selector) || ischar(selector)
        selectorLower = lower(char(selector));
        match = strcmpi(names, string(selectorLower));
        if ~any(match)
            match = contains(lower(names), selectorLower);
        end
        idxCandidates = find(match);
        if isempty(idxCandidates)
            error('run_phase3_compare_models:ModelNotFound', ...
                'Model "%s" not found in the selected model set.', selector);
        end
        chosenIdx = idxCandidates(1);
    else
        error('run_phase3_compare_models:UnsupportedSelector', ...
            'Unsupported model selector of type %s.', class(selector));
    end

    if isempty(chosenIdx) || chosenIdx < 1 || chosenIdx > numel(models)
        error('run_phase3_compare_models:IndexOutOfRange', ...
            'Model index %d is out of range (1..%d).', chosenIdx, numel(models));
    end

    if ~isnan(opts.excludeIndex) && chosenIdx == opts.excludeIndex
        % If user explicitly requests the same index we excluded, pick next available
        remaining = setdiff(indices, opts.excludeIndex, 'stable');
        if isempty(remaining)
            error('run_phase3_compare_models:InsufficientModels', ...
                'Need at least two distinct models to compare.');
        end
        chosenIdx = remaining(1);
    end

    modelInfo = struct('entry', models(chosenIdx), 'index', chosenIdx);
    label = names(chosenIdx);
    if strlength(label) == 0
        label = string(sprintf('Model%d', chosenIdx));
    end
end

function value = getfield_or(S, fieldName, defaultValue)
    if isstruct(S) && isfield(S, fieldName)
        value = S.(fieldName);
    else
        value = defaultValue;
    end
end

function display_summary(summary)
    fprintf('\nPaired t-test on %d observations (%s mode)\n', summary.numSamples, summary.mode);
    fprintf('  %s mean: %.4f (std %.4f)\n', summary.labels(1), summary.meanA, summary.stdA);
    fprintf('  %s mean: %.4f (std %.4f)\n', summary.labels(2), summary.meanB, summary.stdB);
    fprintf('  Mean difference (%s - %s): %.4f\n', summary.labels(1), summary.labels(2), summary.meanDifference);
    fprintf('  t(%d) = %.4f, p = %.4g (alpha = %.3f)\n', summary.df, summary.tstat, summary.pValue, summary.alpha);
    fprintf('  95%% CI: [%.4f, %.4f]\n', summary.confidenceInterval(1), summary.confidenceInterval(2));
    if summary.h
        fprintf('  Result: Significant difference detected.\n');
    else
        fprintf('  Result: No significant difference detected.\n');
    end
    fprintf('  Cohen''s d: %.4f\n\n', summary.cohensD);
end
