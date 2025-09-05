function visualize_fold_metrics(P, opts)
%VISUALIZE_FOLD_METRICS Plot outer fold metrics from Phase 2 results.
%
%   VISUALIZE_FOLD_METRICS(P, opts) loads the latest Phase 2 results file
%   and plots the performance metrics of a selected pipeline across the
%   outer cross-validation folds.
%
%   P    - project paths struct from SETUP_PROJECT_PATHS (optional)
%   opts - plotting options from PLOT_SETTINGS (optional)

    if nargin < 1 || isempty(P)
        P = setup_project_paths(pwd);
    end
    if nargin < 2 || isempty(opts)
        opts = plot_settings();
    end

    phase2ResultsDir = fullfile(P.resultsPath, 'Phase2');
    files = dir(fullfile(phase2ResultsDir, '*_Phase2_AllPipelineResults.mat'));
    if isempty(files)
        error('No Phase 2 results file found in %s.', phase2ResultsDir);
    end
    [~, idx] = sort([files.datenum], 'descend');
    latestFile = fullfile(files(idx(1)).folder, files(idx(1)).name);
    data = load(latestFile);

    if isfield(data, 'resultsPerPipeline')
        resultsCell = data.resultsPerPipeline;
    elseif isfield(data, 'currentStrategyPipelinesResults')
        resultsCell = data.currentStrategyPipelinesResults;
    else
        error('Phase 2 results file missing expected results variable.');
    end

    if isfield(data, 'pipelines')
        pipelineNames = cellfun(@(p) p.name, data.pipelines, 'UniformOutput', false);
    else
        pipelineNames = cellfun(@(s) s.pipelineConfig.name, resultsCell, 'UniformOutput', false);
    end

    fprintf('\nAvailable pipelines:\n');
    for i = 1:numel(pipelineNames)
        fprintf(' %d - %s\n', i, pipelineNames{i});
    end
    usr = input('Select pipeline index to plot (default=1): ','s');
    if isempty(usr)
        selIdx = 1;
    else
        selIdx = str2double(usr);
        if isnan(selIdx) || selIdx < 1 || selIdx > numel(resultsCell)
            selIdx = 1;
        end
    end

    selResult = resultsCell{selIdx};
    if isfield(selResult, 'outerFoldMetrics_raw')
        outerMetrics = selResult.outerFoldMetrics_raw;
    elseif isfield(selResult, 'outerFoldMetrics')
        outerMetrics = selResult.outerFoldMetrics;
    else
        error('Selected pipeline result lacks outer fold metrics.');
    end

    if isfield(data, 'metricNames')
        metricNames = data.metricNames;
    else
        metricNames = arrayfun(@(i) sprintf('Metric %d', i), 1:size(outerMetrics,2), 'UniformOutput', false);
    end

    numMetrics = numel(metricNames);
    numFolds = size(outerMetrics, 1);
    figure('Name', 'Phase 2 Outer Fold Metrics', 'Position', [100, 100, 800, 400]);
    t = tiledlayout(ceil(numMetrics/2), 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, sprintf('Outer Fold Metrics - %s', pipelineNames{selIdx}), 'FontSize', opts.plotFontSize+2, 'FontWeight', 'bold');

    for m = 1:numMetrics
        ax = nexttile;
        plot(ax, 1:numFolds, outerMetrics(:, m), '-o', 'Color', opts.colorCV, 'LineWidth', 1.5);
        xlabel(ax, 'Fold Index', 'FontSize', opts.plotFontSize-1);
        ylabel(ax, strrep(metricNames{m}, '_', ' '), 'FontSize', opts.plotFontSize-1);
        xlim(ax, [1 numFolds]);
        ylim(ax, [0 1]);
        grid(ax, 'on');
    end
end
