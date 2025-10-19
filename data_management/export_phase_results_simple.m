function export_phase_results_simple(varargin)
%EXPORT_PHASE_RESULTS_SIMPLE Export Phase 2 and Phase 3 results to CSV/JSON
%   Simplified version that works with the actual data structure in your files.
%
%   Usage:
%       export_phase_results_simple('projectRoot', pwd)
%       export_phase_results_simple('projectRoot', pwd, ...
%                                   'phase2File', 'results\Phase2\file.mat', ...
%                                   'phase3File', 'results\Phase3\file.mat')

    % Parse inputs
    p = inputParser;
    addParameter(p, 'projectRoot', pwd, @ischar);
    addParameter(p, 'phase2File', '', @ischar);
    addParameter(p, 'phase3File', '', @ischar);
    addParameter(p, 'exportDir', '', @ischar);
    parse(p, varargin{:});
    
    projectRoot = p.Results.projectRoot;
    phase2File = p.Results.phase2File;
    phase3File = p.Results.phase3File;
    exportDir = p.Results.exportDir;
    
    % Set default export directory
    if isempty(exportDir)
        exportDir = fullfile(projectRoot, 'results', 'Exports');
    end
    if ~isfolder(exportDir)
        mkdir(exportDir);
    end
    
    % Find Phase 2 file if not specified
    if isempty(phase2File)
        phase2Dir = fullfile(projectRoot, 'results', 'Phase2');
        files = dir(fullfile(phase2Dir, '*Phase2*.mat'));
        if ~isempty(files)
            [~, idx] = max([files.datenum]);
            phase2File = fullfile(phase2Dir, files(idx).name);
        end
    end
    
    % Find Phase 3 file if not specified
    if isempty(phase3File)
        phase3Dir = fullfile(projectRoot, 'results', 'Phase3');
        files = dir(fullfile(phase3Dir, '*Phase3*.mat'));
        if ~isempty(files)
            [~, idx] = max([files.datenum]);
            phase3File = fullfile(phase3Dir, files(idx).name);
        end
    end
    
    fprintf('Export configuration:\n');
    fprintf('  Project root: %s\n', projectRoot);
    fprintf('  Phase 2 file: %s\n', phase2File);
    fprintf('  Phase 3 file: %s\n', phase3File);
    fprintf('  Export dir:   %s\n', exportDir);
    fprintf('\n');
    
    % =====================================================================
    % Export Phase 2 results
    % =====================================================================
    if ~isempty(phase2File) && isfile(phase2File)
        fprintf('Processing Phase 2 results...\n');
        try
            data2 = load(phase2File);
            
            if isfield(data2, 'resultsPerPipeline') && iscell(data2.resultsPerPipeline)
                phase2Table = export_phase2_to_table(data2, phase2File, projectRoot);
                if ~isempty(phase2Table)
                    csvFile = fullfile(exportDir, 'phase2_pipeline_leaderboard.csv');
                    writetable(phase2Table, csvFile);
                    fprintf('  Written: %s\n', csvFile);
                end
            else
                warning('Phase 2 file does not contain expected resultsPerPipeline structure.');
            end
        catch ME
            warning('Error processing Phase 2 file: %s', ME.message);
        end
    else
        fprintf('No Phase 2 file found or specified.\n');
    end
    
    % =====================================================================
    % Export Phase 3 results
    % =====================================================================
    if ~isempty(phase3File) && isfile(phase3File)
        fprintf('Processing Phase 3 results...\n');
        try
            data3 = load(phase3File);
            
            % Handle the simple structure: results array + bestModelInfo
            if isfield(data3, 'results') && isstruct(data3.results)
                [phase3Table, bestTable] = export_phase3_simple_to_tables(data3, phase3File, projectRoot);
                
                if ~isempty(phase3Table)
                    csvFile = fullfile(exportDir, 'phase3_model_metrics.csv');
                    writetable(phase3Table, csvFile);
                    fprintf('  Written: %s\n', csvFile);
                end
                
                if ~isempty(bestTable)
                    csvFile = fullfile(exportDir, 'phase3_best_model.csv');
                    writetable(bestTable, csvFile);
                    fprintf('  Written: %s\n', csvFile);
                end
            else
                warning('Phase 3 file does not contain expected results structure.');
            end
        catch ME
            warning('Error processing Phase 3 file: %s', ME.message);
        end
    else
        fprintf('No Phase 3 file found or specified.\n');
    end
    
    fprintf('\nExports completed. Output directory: %s\n', exportDir);
end

function tbl = export_phase2_to_table(data, filePath, projectRoot)
    % Extract Phase 2 results into a table
    rows = struct([]);
    
    metricNames = {};
    if isfield(data, 'metricNames')
        metricNames = cellstr(data.metricNames);
    end
    
    for p = 1:numel(data.resultsPerPipeline)
        res = data.resultsPerPipeline{p};
        if isempty(res) || ~isfield(res, 'pipelineConfig')
            continue;
        end
        
        pipe = res.pipelineConfig;
        pipeName = get_field_or(pipe, 'name', sprintf('Pipeline%d', p));
        
        row = struct();
        row.pipeline_name = string(pipeName);
        row.feature_selection = string(get_field_or(pipe, 'feature_selection_method', ''));
        row.classifier = string(get_field_or(pipe, 'classifier', ''));
        row.results_file = string(make_relative(filePath, projectRoot));
        
        % Add metrics
        if isfield(res, 'outerFoldMetrics_mean')
            metrics = res.outerFoldMetrics_mean;
            for m = 1:min(numel(metrics), numel(metricNames))
                metricName = matlab.lang.makeValidName(['CV_' metricNames{m}]);
                row.(metricName) = metrics(m);
            end
        end
        
        if isempty(rows)
            rows = row;
        else
            rows(end+1) = row;
        end
    end
    
    if ~isempty(rows)
        tbl = struct2table(rows);
    else
        tbl = table();
    end
end

function [tbl, bestTbl] = export_phase3_simple_to_tables(data, filePath, projectRoot)
    % Extract Phase 3 results (simple structure) into tables
    rows = struct([]);
    
    % Process each model in the results array
    for i = 1:numel(data.results)
        model = data.results(i);
        
        row = struct();
        row.model_name = string(get_field_or(model, 'name', sprintf('Model%d', i)));
        row.model_file = string(make_relative(get_field_or(model, 'modelFile', ''), projectRoot));
        row.roc_file = string(make_relative(get_field_or(model, 'rocFile', ''), projectRoot));
        row.results_file = string(make_relative(filePath, projectRoot));
        
        % Add test set metrics
        if isfield(model, 'metrics') && isstruct(model.metrics)
            metrics = model.metrics;
            metricFields = fieldnames(metrics);
            for m = 1:numel(metricFields)
                val = metrics.(metricFields{m});
                if isnumeric(val) && isscalar(val)
                    row.(metricFields{m}) = val;
                end
            end
        end
        
        % Add CV metrics if available
        if isfield(model, 'CV_Metrics') && isnumeric(model.CV_Metrics)
            cvMetrics = model.CV_Metrics;
            for m = 1:numel(cvMetrics)
                row.(sprintf('CV_Metric%d', m)) = cvMetrics(m);
            end
        end
        
        if isempty(rows)
            rows = row;
        else
            rows(end+1) = row;
        end
    end
    
    if ~isempty(rows)
        tbl = struct2table(rows);
    else
        tbl = table();
    end
    
    % Create best model table
    bestRow = struct();
    if isfield(data, 'bestModelInfo') && isstruct(data.bestModelInfo)
        best = data.bestModelInfo;
        bestRow.model_name = string(get_field_or(best, 'name', ''));
        bestRow.model_file = string(make_relative(get_field_or(best, 'modelFile', ''), projectRoot));
        bestRow.roc_file = string(make_relative(get_field_or(best, 'rocFile', ''), projectRoot));
        
        if isfield(best, 'metrics') && isstruct(best.metrics)
            metrics = best.metrics;
            metricFields = fieldnames(metrics);
            for m = 1:numel(metricFields)
                val = metrics.(metricFields{m});
                if isnumeric(val) && isscalar(val)
                    bestRow.(metricFields{m}) = val;
                end
            end
        end
        
        bestTbl = struct2table(bestRow);
    else
        bestTbl = table();
    end
end

function val = get_field_or(S, fieldName, defaultValue)
    if isstruct(S) && isfield(S, fieldName)
        val = S.(fieldName);
    else
        val = defaultValue;
    end
end

function rel = make_relative(fullPath, basePath)
    if isempty(fullPath) || ~ischar(fullPath)
        rel = '';
        return;
    end
    
    % Normalize paths
    fullPath = char(fullPath);
    basePath = char(basePath);
    
    % Try to make relative
    if startsWith(fullPath, basePath)
        rel = fullPath(length(basePath)+1:end);
        if startsWith(rel, filesep)
            rel = rel(2:end);
        end
    else
        rel = fullPath;
    end
end