function visualize_project_summary(cfg, opts)
%VISUALIZE_PROJECT_SUMMARY Generate summary plots for Phases 2-4.
%
%   VISUALIZE_PROJECT_SUMMARY(cfg, opts) loads the latest Phase 2 and Phase 3
%   results and creates a spider plot comparing cross-validation and test
%   performance of the best pipeline.
%   cfg  - configuration struct from CONFIGURE_CFG (optional)
%   opts - plotting options from PLOT_SETTINGS (optional)

    if nargin < 1 || isempty(cfg)
        cfg = configure_cfg();
    end
    if nargin < 2 || isempty(opts)
        opts = plot_settings();
    end

    fprintf('GENERATING PROJECT VISUALIZATIONS (Phases 2-4) - %s\n', string(datetime('now')));

    P = setup_project_paths(cfg.projectRoot);
    figuresPath_output = fullfile(P.figuresPath, 'ProjectSummaryFigures');
    if ~isfolder(figuresPath_output), mkdir(figuresPath_output); end
    dateStrForFilenames = opts.datePrefix;

    if exist(P.helperFunPath, 'dir'), addpath(P.helperFunPath); end
    if exist(fullfile(P.projectRoot, 'plotting'), 'dir'), addpath(fullfile(P.projectRoot, 'plotting')); end
    if exist('spider_plot_R2019b', 'file') ~= 2
        error('Required plotting helper "spider_plot_R2019b.m" is missing from the MATLAB path.');
    end

    colorCV   = opts.colorCV;
    colorTest = opts.colorTest;

    %% 1. Load Phase 2 and Phase 3 Results
    strategy = cfg.outlierStrategy;
    p3_files = dir(fullfile(P.resultsPath, 'Phase3', sprintf('*_Phase3_ComparisonResults_Strat_%s.mat', strategy)));
    if isempty(p3_files)
        error('No Phase 3 results file found for strategy %s. Run Phase 3 first.', strategy);
    end
    [~,idxSortP3] = sort([p3_files.datenum],'descend');
    p3_data = load(fullfile(p3_files(idxSortP3(1)).folder, p3_files(idxSortP3(1)).name), 'bestModelInfo');
    bestPipelineName = p3_data.bestModelInfo.name;
    bestPipelineTestMetrics = p3_data.bestModelInfo.metrics;

    p2_files = dir(fullfile(P.resultsPath, 'Phase2', sprintf('*_Phase2_AllPipelineResults_Strat_%s.mat', strategy)));
    if isempty(p2_files)
        error('No Phase 2 results file found for strategy %s. Run Phase 2 first.', strategy);
    end
    [~,idxSortP2] = sort([p2_files.datenum],'descend');
    p2_data = load(fullfile(p2_files(idxSortP2(1)).folder, p2_files(idxSortP2(1)).name), 'currentStrategyPipelinesResults', 'pipelines', 'metricNames');

    bestPipelineCVMetrics = [];
    for i = 1:length(p2_data.pipelines)
        if strcmpi(p2_data.pipelines{i}.name, bestPipelineName)
            bestPipelineCVMetrics = p2_data.currentStrategyPipelinesResults{i}.outerFoldMetrics_mean;
            break;
        end
    end
    if isempty(bestPipelineCVMetrics)
        error('Could not find CV metrics for the best pipeline (%s) in the Phase 2 results file.', bestPipelineName);
    end

    %% 2. Spider Plot
    spider_metrics_to_plot = {'Accuracy','Sensitivity_WHO3','Specificity_WHO1','PPV_WHO3','F1_WHO3','AUC'};
    spider_axes_labels = strrep(spider_metrics_to_plot, '_', ' ');
    metricNames_p2 = p2_data.metricNames;
    P_spider = zeros(2, numel(spider_metrics_to_plot));
    for i = 1:numel(spider_metrics_to_plot)
        idx = find(strcmpi(metricNames_p2, spider_metrics_to_plot{i}));
        if ~isempty(idx)
            P_spider(1,i) = bestPipelineCVMetrics(idx);
        end
        if isfield(bestPipelineTestMetrics, spider_metrics_to_plot{i})
            P_spider(2,i) = bestPipelineTestMetrics.(spider_metrics_to_plot{i});
        end
    end

    figSpider = figure('Name', ['Performance Profile: ' bestPipelineName], 'Position', [100, 100, 700, 600]);
    axesLimitsSpider = repmat([0.5; 1.0], 1, numel(spider_metrics_to_plot));
    spider_plot_R2019b(P_spider, 'AxesLabels', spider_axes_labels, 'AxesLimits', axesLimitsSpider, ...
        'AxesInterval',5,'AxesPrecision',2,'FillOption','on','FillTransparency',[0.2,0.1], ...
        'Color',[colorCV; colorTest],'LineWidth',2.5,'Marker',{'o','s'},'MarkerSize',80);
    title({sprintf('Performance Profile: %s Pipeline', bestPipelineName); sprintf('(Outlier Strategy: %s)', strategy)}, 'FontSize', 14);
    legend({'Mean Cross-Validation','Final Test Set'},'Location','southoutside','FontSize',12);

    outBase = fullfile(figuresPath_output, sprintf('%s_P_Summary_SpiderPlot_%s', dateStrForFilenames, bestPipelineName));
    savefig(figSpider,[outBase '.fig']);
    exportgraphics(figSpider,[outBase '.tiff'],'Resolution',300);
    fprintf('Performance Profile spider plot saved to: %s.tiff\n', outBase);
end
