function export_phase_results_to_csv_json(cfg)
%EXPORT_PHASE_RESULTS_TO_CSV_JSON Create human and machine-friendly exports.
%   EXPORT_PHASE_RESULTS_TO_CSV_JSON(CFG) loads the most recent Phase 2 and
%   Phase 3 MATLAB result files and writes summary CSV tables together with a
%   comprehensive JSON bundle for downstream analysis. Optional fields in CFG:
%
%     projectRoot           - Repository root (defaults to get_project_root()).
%     phase2ResultsFile     - Explicit path to a Phase 2 results MAT-file.
%     phase3ResultsFile     - Explicit path to a Phase 3 results MAT-file.
%     exportRoot            - Destination folder for exports (defaults to
%                             results/Exports relative to projectRoot).
%
%   The function generates:
%     * Phase 2 leaderboard table (CSV)
%     * Phase 3 variant-by-model metrics table (CSV)
%     * Phase 3 best-model summary table (CSV)
%     * JSON bundle containing the key information from both phases
%
%   Example:
%       cfg = struct('projectRoot', '/path/to/repo');
%       export_phase_results_to_csv_json(cfg);
%
%   See also: RUN_PHASE2_MODEL_SELECTION, RUN_PHASE3_FINAL_EVALUATION
%
%   Date: 2025-07-01
%
arguments
    cfg.projectRoot string = string(get_project_root())
    cfg.phase2ResultsFile string = ""
    cfg.phase3ResultsFile string = ""
    cfg.exportRoot string = ""
end

projectRoot = char(cfg.projectRoot);

% Ensure helper functions are on the path and common directories exist
setup_project_paths(projectRoot);

if strlength(cfg.exportRoot) > 0
    exportRoot = char(cfg.exportRoot);
else
    exportRoot = fullfile(projectRoot, 'results', 'Exports');
end
if ~isfolder(exportRoot); mkdir(exportRoot); end

phase2Dir = fullfile(projectRoot, 'results', 'Phase2');
phase3Dir = fullfile(projectRoot, 'results', 'Phase3');

% ---------------------------------------------------------------------
% Phase 2 exports
phase2Files = locate_result_files(phase2Dir, cfg.phase2ResultsFile, '*_Phase2_*_AllPipelineResults.mat');

phase2Rows = struct([]);
phase2JsonEntries = struct([]);

for f = 1:numel(phase2Files)
    filePath = phase2Files{f};
    data = load(filePath);
    if ~isfield(data, 'resultsPerPipeline') || isempty(data.resultsPerPipeline)
        warning('Phase 2 file %s does not contain resultsPerPipeline. Skipping.', filePath);
        continue;
    end
    if ~iscell(data.resultsPerPipeline)
        warning('Phase 2 file %s has unexpected resultsPerPipeline type (%s). Skipping.', filePath, class(data.resultsPerPipeline));
        continue;
    end

    metricNames = getfield_or(data, 'metricNames', compose("Metric%d", 1:size(data.resultsPerPipeline{1}.outerFoldMetrics_mean, 2))); %#ok<GFLD>
    datasetInfo = getfield_or(data, 'ds', struct('id', '', 'description', ''));
    numOuterFolds = getfield_or(data, 'numOuterFolds', NaN);
    numInnerFolds = getfield_or(data, 'numInnerFolds', NaN);

    for p = 1:numel(data.resultsPerPipeline)
        res = data.resultsPerPipeline{p};
        if isempty(res) || ~isfield(res, 'pipelineConfig')
            continue;
        end
        pipe = res.pipelineConfig;
        pipeName = getfield_or(pipe, 'name', sprintf('Pipeline%d', p));
        aggHyper = aggregate_best_hyperparams(getfield_or(res, 'outerFoldBestHyperparams', {}));
        hyperSummary = format_struct_compact(aggHyper);

        metricsMean = getfield_or(res, 'outerFoldMetrics_mean', NaN(1, numel(metricNames)));
        metricsStruct = metrics_vector_to_struct(metricsMean, metricNames, 'CV_');

        row = struct();
        row.dataset_id = string(getfield_or(datasetInfo, 'id', ''));
        row.dataset_description = string(getfield_or(datasetInfo, 'description', ''));
        row.pipeline_name = string(pipeName);
        row.feature_selection = string(getfield_or(pipe, 'feature_selection_method', ''));
        row.classifier = string(getfield_or(pipe, 'classifier', ''));
        row.hyperparameters = string(hyperSummary);
        row.num_outer_folds = numOuterFolds;
        row.num_inner_folds = numInnerFolds;
        row.results_file = string(make_relative_path(filePath, projectRoot));
        row.final_model_file = string(make_relative_path(getfield_or(res, 'finalModelFile', ''), projectRoot));

        metricFields = fieldnames(metricsStruct);
        for m = 1:numel(metricFields)
            row.(metricFields{m}) = metricsStruct.(metricFields{m});
        end

        if isempty(phase2Rows)
            phase2Rows = row;
        else
            phase2Rows(end+1) = row; %#ok<AGROW>
        end

        entry = struct();
        entry.dataset = datasetInfo;
        entry.pipelineConfig = pipe;
        entry.resultsFile = make_relative_path(filePath, projectRoot);
        entry.finalModelFile = make_relative_path(getfield_or(res, 'finalModelFile', ''), projectRoot);
        entry.numOuterFolds = numOuterFolds;
        entry.numInnerFolds = numInnerFolds;
        entry.crossValidation = struct();
        entry.crossValidation.metricsMean = metrics_vector_to_struct(metricsMean, metricNames, '');
        entry.crossValidation.metricsPerFold = fold_metrics_to_structs(getfield_or(res, 'outerFoldMetrics_raw', []), metricNames);
        entry.crossValidation.metricNames = metricNames;
        entry.crossValidation.bestHyperparameters = hyperparams_cell_to_structs(getfield_or(res, 'outerFoldBestHyperparams', {}));
        entry.crossValidation.aggregatedHyperparameters = aggHyper;

        if isempty(phase2JsonEntries)
            phase2JsonEntries = entry;
        else
            phase2JsonEntries(end+1) = entry; %#ok<AGROW>
        end
    end
end

if ~isempty(phase2Rows)
    phase2Table = struct2table(phase2Rows);
    writetable(phase2Table, fullfile(exportRoot, 'phase2_pipeline_leaderboard.csv'));
end

% ---------------------------------------------------------------------
% Phase 3 exports
phase3Files = locate_result_files(phase3Dir, cfg.phase3ResultsFile, '*_Phase3_ParallelComparisonResults.mat');

phase3Rows = struct([]);
bestRows = struct([]);
phase3JsonVariants = struct([]);
bestJson = struct([]);

for f = 1:numel(phase3Files)
    filePath = phase3Files{f};
    data = load(filePath);
    if ~isfield(data, 'resultsByVariant') || isempty(data.resultsByVariant)
        warning('Phase 3 file %s does not contain resultsByVariant. Skipping.', filePath);
        continue;
    end

    variants = data.resultsByVariant;
    modelSetsMeta = getfield_or(data, 'modelSets', struct([]));
    modelSetsMeta = ensure_cell(modelSetsMeta);

    for v = 1:numel(variants)
        variant = variants(v);
        variantEntry = struct();
        variantEntry.id = getfield_or(variant, 'id', sprintf('Variant%d', v));
        variantEntry.description = getfield_or(variant, 'description', '');
        variantEntry.modelSets = struct([]);

        modelSets = getfield_or(variant, 'modelSets', struct([]));
        for s = 1:numel(modelSets)
            ms = modelSets(s);
            modelSetEntry = struct();
            modelSetEntry.id = getfield_or(ms, 'modelSetID', sprintf('ModelSet%d', s));
            modelSetEntry.description = getfield_or(ms, 'modelSetDescription', '');
            metaIdx = find_model_set_meta(modelSetsMeta, modelSetEntry.id);
            if metaIdx > 0
                meta = modelSetsMeta{metaIdx};
                modelSetEntry.modelsDir = make_relative_path(getfield_or(meta, 'modelsDir', ''), projectRoot);
                modelSetEntry.resultsDir = make_relative_path(getfield_or(meta, 'resultsDir', ''), projectRoot);
                cvMetricNames = getfield_or(meta, 'metricNames', {});
            else
                modelSetEntry.modelsDir = '';
                modelSetEntry.resultsDir = '';
                cvMetricNames = {};
            end

            models = getfield_or(ms, 'models', struct([]));
            modelEntries = struct([]);
            for mIdx = 1:numel(models)
                mdl = models(mIdx);
                metricsStruct = ensure_struct(mdl.metrics);

                row = struct();
                row.variant_id = string(variantEntry.id);
                row.variant_description = string(variantEntry.description);
                row.model_set_id = string(modelSetEntry.id);
                row.model_set_description = string(modelSetEntry.description);
                row.pipeline_name = string(getfield_or(mdl, 'name', sprintf('Model%d', mIdx)));
                row.model_file = string(make_relative_path(getfield_or(mdl, 'modelFile', ''), projectRoot));
                row.roc_file = string(make_relative_path(getfield_or(mdl, 'rocFile', ''), projectRoot));
                row.results_file = string(make_relative_path(filePath, projectRoot));

                metricFields = fieldnames(metricsStruct);
                for mf = 1:numel(metricFields)
                    value = metricsStruct.(metricFields{mf});
                    if isnumeric(value) && isscalar(value)
                        row.(metricFields{mf}) = value;
                    else
                        row.(metricFields{mf}) = string(value);
                    end
                end

                cvMetrics = getfield_or(mdl, 'CV_Metrics', []);
                if ~isempty(cvMetrics) && ~isempty(cvMetricNames)
                    cvStruct = metrics_vector_to_struct(cvMetrics(:)', cvMetricNames, 'CV_');
                    cvFields = fieldnames(cvStruct);
                    for cf = 1:numel(cvFields)
                        row.(cvFields{cf}) = cvStruct.(cvFields{cf});
                    end
                end

                if isempty(phase3Rows)
                    phase3Rows = row;
                else
                    phase3Rows(end+1) = row; %#ok<AGROW>
                end

                modelEntry = struct();
                modelEntry.name = getfield_or(mdl, 'name', '');
                modelEntry.metrics = metricsStruct;
                modelEntry.modelFile = make_relative_path(getfield_or(mdl, 'modelFile', ''), projectRoot);
                modelEntry.rocFile = make_relative_path(getfield_or(mdl, 'rocFile', ''), projectRoot);
                modelEntry.scores = getfield_or(mdl, 'scores', []);
                modelEntry.predicted = getfield_or(mdl, 'predicted', []);
                modelEntry.probeTable = table_to_struct_array(getfield_or(mdl, 'probeTable', table()));
                modelEntry.probeMetrics = ensure_struct(getfield_or(mdl, 'probeMetrics', struct()));
                if ~isempty(cvMetrics) && ~isempty(cvMetricNames)
                    modelEntry.crossValidation = metrics_vector_to_struct(cvMetrics(:)', cvMetricNames, '');
                else
                    modelEntry.crossValidation = struct();
                end

                if isempty(modelEntries)
                    modelEntries = modelEntry;
                else
                    modelEntries(end+1) = modelEntry; %#ok<AGROW>
                end
            end
            modelSetEntry.models = modelEntries;

            if isempty(variantEntry.modelSets)
                variantEntry.modelSets = modelSetEntry;
            else
                variantEntry.modelSets(end+1) = modelSetEntry; %#ok<AGROW>
            end
        end

        if isempty(phase3JsonVariants)
            phase3JsonVariants = variantEntry;
        else
            phase3JsonVariants(end+1) = variantEntry; %#ok<AGROW>
        end
    end

    if isfield(data, 'bestModelInfo') && ~isempty(data.bestModelInfo)
        bestEntries = data.bestModelInfo;
        for b = 1:numel(bestEntries)
            bmi = bestEntries(b);
            row = struct();
            row.variant_id = string(getfield_or(bmi, 'variantID', ''));
            row.model_set_id = string(getfield_or(bmi, 'modelSetID', ''));
            row.pipeline_name = string(getfield_or(bmi, 'modelName', ''));
            row.model_file = string(make_relative_path(getfield_or(bmi, 'modelFile', ''), projectRoot));
            row.results_file = string(make_relative_path(filePath, projectRoot));

            metricsStruct = ensure_struct(getfield_or(bmi, 'metrics', struct()));
            metricFields = fieldnames(metricsStruct);
            for mf = 1:numel(metricFields)
                value = metricsStruct.(metricFields{mf});
                if isnumeric(value) && isscalar(value)
                    row.(metricFields{mf}) = value;
                else
                    row.(metricFields{mf}) = string(value);
                end
            end

            if isempty(bestRows)
                bestRows = row;
            else
                bestRows(end+1) = row; %#ok<AGROW>
            end

            bmiEntry = struct();
            bmiEntry.variantID = getfield_or(bmi, 'variantID', '');
            bmiEntry.modelSetID = getfield_or(bmi, 'modelSetID', '');
            bmiEntry.modelName = getfield_or(bmi, 'modelName', '');
            bmiEntry.metrics = metricsStruct;
            bmiEntry.modelFile = make_relative_path(getfield_or(bmi, 'modelFile', ''), projectRoot);
            if isempty(bestJson)
                bestJson = bmiEntry;
            else
                bestJson(end+1) = bmiEntry; %#ok<AGROW>
            end
        end
    end
end

if ~isempty(phase3Rows)
    phase3Table = struct2table(phase3Rows);
    writetable(phase3Table, fullfile(exportRoot, 'phase3_variant_model_metrics.csv'));
end

if ~isempty(bestRows)
    bestTable = struct2table(bestRows);
    writetable(bestTable, fullfile(exportRoot, 'phase3_best_models.csv'));
end

jsonBundle = struct();
jsonBundle.generatedAt = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
jsonBundle.projectRoot = make_relative_path(projectRoot, projectRoot);
jsonBundle.phase2 = phase2JsonEntries;
jsonBundle.phase3 = struct('variants', phase3JsonVariants, 'bestModels', bestJson);

jsonText = encode_json(jsonBundle);
fid = fopen(fullfile(exportRoot, 'phase_metrics_bundle.json'), 'w');
if fid == -1
    error('Failed to open JSON export for writing.');
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonText);
clear cleaner;

fprintf('Exports written to %s
', exportRoot);

end

function metricStruct = metrics_vector_to_struct(values, metricNames, prefix)
    if nargin < 3; prefix = ''; end
    metricStruct = struct();
    if isempty(values) || isempty(metricNames)
        return;
    end
    metricNames = cellstr(metricNames);
    for i = 1:min(numel(metricNames), numel(values))
        name = matlab.lang.makeValidName(sprintf('%s%s', prefix, metricNames{i}));
        metricStruct.(name) = values(i);
    end
end

function foldStruct = fold_metrics_to_structs(raw, metricNames)
    foldStruct = struct([]);
    if isempty(raw)
        return;
    end
    for i = 1:size(raw, 1)
        entry = metrics_vector_to_struct(raw(i, :), metricNames, '');
        entry.foldIndex = i;
        if isempty(foldStruct)
            foldStruct = entry;
        else
            foldStruct(end+1) = entry; %#ok<AGROW>
        end
    end
end

function structs = hyperparams_cell_to_structs(hyperCell)
    structs = struct([]);
    if isempty(hyperCell)
        return;
    end
    for i = 1:numel(hyperCell)
        entry = struct('foldIndex', i, 'hyperparameters', hyperCell{i});
        if isempty(structs)
            structs = entry;
        else
            structs(end+1) = entry; %#ok<AGROW>
        end
    end
end

function text = format_struct_compact(S)
    if isempty(S) || ~isstruct(S)
        text = "";
        return;
    end
    fields = fieldnames(S);
    parts = strings(1, numel(fields));
    for i = 1:numel(fields)
        val = S.(fields{i});
        parts(i) = sprintf('%s=%s', fields{i}, format_value(val)); %#ok<SPRINTFC>
    end
    text = strjoin(parts, '; ');
end

function text = format_value(val)
    if isnumeric(val)
        text = strtrim(regexprep(sprintf(' %g', val), '^\s', ''));
    elseif isstring(val)
        text = char(val);
    elseif ischar(val)
        text = val;
    else
        text = jsonencode(val);
    end
end

function rel = make_relative_path(targetPath, basePath)
    if nargin < 2 || isempty(basePath)
        basePath = pwd;
    end
    if isempty(targetPath)
        rel = "";
        return;
    end
    targetPath = char(targetPath);
    basePath = char(basePath);
    if startsWith(targetPath, basePath)
        rel = eraseBetween(targetPath, 1, length(basePath));
        if startsWith(rel, filesep)
            rel = rel(2:end);
        end
        if isempty(rel)
            rel = ".";
        end
    else
        rel = targetPath;
    end
    rel = string(rel);
end

function out = ensure_struct(val)
    if isempty(val)
        out = struct();
    elseif istable(val)
        out = table2struct(val);
    elseif isstruct(val)
        out = val;
    else
        out = struct('value', val);
    end
end

function arr = table_to_struct_array(tbl)
    if istable(tbl)
        if height(tbl) == 0
            arr = struct([]);
        else
            arr = table2struct(tbl);
        end
    else
        arr = struct([]);
    end
end

function idx = find_model_set_meta(modelSetsMeta, identifier)
    idx = 0;
    if isempty(modelSetsMeta)
        return;
    end
    for i = 1:numel(modelSetsMeta)
        meta = modelSetsMeta{i};
        if isstruct(meta) && isfield(meta, 'id') && strcmp(meta.id, identifier)
            idx = i;
            return;
        end
    end
end

function files = locate_result_files(baseDir, explicitFile, pattern)
    files = {};
    if strlength(explicitFile) > 0
        if isfile(explicitFile)
            files = {char(explicitFile)};
        else
            warning('Specified file %s does not exist.', explicitFile);
        end
        return;
    end
    if ~isfolder(baseDir)
        warning('Directory %s does not exist.', baseDir);
        return;
    end
    listing = dir(fullfile(baseDir, '**', pattern));
    if isempty(listing)
        warning('No files matching %s found under %s.', pattern, baseDir);
        return;
    end
    [~, order] = sort([listing.datenum], 'descend');
    listing = listing(order);
    files = arrayfun(@(f) fullfile(f.folder, f.name), listing, 'UniformOutput', false);
end

function value = getfield_or(S, fieldName, defaultValue)
    if isstruct(S) && isfield(S, fieldName)
        value = S.(fieldName);
    else
        value = defaultValue;
    end
end

function cellArray = ensure_cell(value)
    if isempty(value)
        cellArray = {};
    elseif iscell(value)
        cellArray = value;
    elseif isstruct(value)
        cellArray = num2cell(value);
    else
        cellArray = {value};
    end
end

function text = encode_json(data)
    try
        text = jsonencode(data, 'PrettyPrint', true);
    catch
        text = jsonencode(data);
    end
end
