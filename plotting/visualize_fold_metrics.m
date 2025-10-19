function visualize_fold_metrics(cfg, opts)
%VISUALIZE_FOLD_METRICS Plot cross-validation fold metrics from Phase 2.
%
%   VISUALIZE_FOLD_METRICS(cfg, opts) loads the most recent Phase 2
%   results file, prompts the user to select a pipeline and produces a bar
%   chart of the outer-fold performance metrics for that pipeline.
%   cfg  - configuration struct from CONFIGURE_CFG (optional)
%   opts - plotting options from PLOT_SETTINGS (optional)
%
%   The plot groups metrics along the x-axis with bars corresponding to each
%   outer fold. Figures are saved to the Phase2/FoldMetrics directory within
%   the project's figures folder.
%
%   Example:
%       visualize_fold_metrics();
%       cfg = configure_cfg('projectRoot', '/path/to/project');
%       visualize_fold_metrics(cfg);
%
%   Date: 2025-06-11

    if nargin < 1 || isempty(cfg)
        cfg = configure_cfg();
    end
    if nargin < 2 || isempty(opts)
        opts = plot_settings();
    end

    P = setup_project_paths(cfg.projectRoot, '', cfg);

    %% Locate latest Phase 2 results file
    p2Dir = fullfile(P.resultsPath, 'Phase2');
    files = dir(fullfile(p2Dir, '*_Phase2_AllPipelineResults.mat'));
    if isempty(files)
        error('No Phase 2 results file found in %s.', p2Dir);
    end
    [~, idx] = sort([files.datenum], 'descend');
    resFile = fullfile(files(idx(1)).folder, files(idx(1)).name);
    S = load(resFile);

    % Determine pipeline results variable name
    if isfield(S, 'resultsPerPipeline')
        pipelineResults = S.resultsPerPipeline;
    elseif isfield(S, 'currentStrategyPipelinesResults')
        pipelineResults = S.currentStrategyPipelinesResults;
    else
        error('No pipeline results found in %s.', resFile);
    end

    if isfield(S, 'pipelines')
        pipelines = S.pipelines;
    elseif isfield(S, 'pipelineConfigs')
        pipelines = S.pipelineConfigs;
    else
        error('No pipeline configurations found in %s.', resFile);
    end

    if isfield(S, 'metricNames')
        metricNames = S.metricNames;
    else
        error('metricNames not found in %s.', resFile);
    end

    pipelineNames = cellfun(@(p) p.name, pipelines, 'UniformOutput', false);
    fprintf('Available pipelines in %s:\n', resFile);
    for i = 1:numel(pipelineNames)
        fprintf(' %d - %s\n', i, pipelineNames{i});
    end
    usr = input(sprintf('Select pipeline (1-%d): ', numel(pipelineNames)), 's');
    selIdx = str2double(usr);
    if isnan(selIdx) || selIdx < 1 || selIdx > numel(pipelineNames)
        selIdx = 1;
        fprintf('Invalid selection. Using pipeline 1 (%s).\n', pipelineNames{1});
    end

    if ~isfield(pipelineResults{selIdx}, 'outerFoldMetrics_raw')
        error('outerFoldMetrics_raw not found for selected pipeline.');
    end
    foldMetrics = pipelineResults{selIdx}.outerFoldMetrics_raw; % numFolds x numMetrics
    numFolds = size(foldMetrics, 1);

    %% Plot
    fig = figure('Name', sprintf('Phase 2 Fold Metrics - %s', pipelineNames{selIdx}));
    bar(foldMetrics');
    set(gca, 'XTick', 1:numel(metricNames), 'XTickLabel', strrep(metricNames, '_', ' '), ...
             'FontSize', opts.plotFontSize);
    xlabel('Metric');
    ylabel('Score');
    legend(arrayfun(@(k) sprintf('Fold %d', k), 1:numFolds, 'UniformOutput', false), ...
           'Location', 'bestoutside');
    title(sprintf('Outer Fold Metrics: %s', pipelineNames{selIdx}), 'Interpreter', 'none');
    xtickangle(45);
    grid on;

    %% Save figure
    outDir = fullfile(P.figuresPath, 'Phase2', 'FoldMetrics');
    if ~isfolder(outDir), mkdir(outDir); end
    outBase = fullfile(outDir, sprintf('%s_P2_FoldMetrics_%s', opts.datePrefix, pipelineNames{selIdx}));
    savefig(fig, [outBase '.fig']);
    exportgraphics(fig, [outBase '.tiff'], 'Resolution', 300);
    fprintf('Fold metrics plot saved to: %s.(fig/tiff)\n', outBase);
end
