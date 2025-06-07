% run_visualize_project_results.m
%
% Script to generate key visualizations for the meningioma FT-IR project.
% MODIFIED to identify the best pipeline from Phase 3 and generate a
% detailed spider plot comparing its CV vs. Test Set performance.
%
% Date: 2025-06-07

%% 0. Initialization
fprintf('GENERATING PROJECT VISUALIZATIONS (Phases 2-4) - %s\n', string(datetime('now')));
clear; clc; close all;

% --- Configuration & Paths ---
cfg = configure_cfg(); % Use helper to get project root and default strategy
P = setup_project_paths(cfg.projectRoot); % Use helper to get all paths
figuresPath_output = fullfile(P.figuresPath, 'ProjectSummaryFigures');
if ~isfolder(figuresPath_output), mkdir(figuresPath_output); end
dateStrForFilenames = string(datetime('now','Format','yyyyMMdd'));

% Add plotting helpers to path
if exist(P.helperFunPath, 'dir'), addpath(P.helperFunPath); end
if exist(fullfile(P.projectRoot, 'plotting'), 'dir'), addpath(fullfile(P.projectRoot, 'plotting')); end
if exist('spider_plot_R2019b', 'file') ~= 2
    error('Required plotting helper "spider_plot_R2019b.m" is missing from the MATLAB path.');
end

% --- Plotting Defaults ---
colorCV = [0.2 0.6 0.2]; % Green for CV
colorTest = [0.8 0.2 0.2]; % Red for Test

%% 1. Load Phase 2 and Phase 3 Results to Find Best Pipeline
fprintf('\n--- 1. Loading Phase 2 & 3 Results to Identify Best Pipeline ---\n');

% --- Load Phase 3 Results to find the name and test performance of the best model ---
strategy = cfg.outlierStrategy;
p3_results_files = dir(fullfile(P.resultsPath, 'Phase3', sprintf('*_Phase3_ComparisonResults_Strat_%s.mat', strategy)));
if isempty(p3_results_files)
    error('No Phase 3 results file found for strategy %s. Run Phase 3 first.', strategy);
end
[~,idxSortP3] = sort([p3_results_files.datenum],'descend');
latest_p3_results_file = fullfile(p3_results_files(idxSortP3(1)).folder, p3_results_files(idxSortP3(1)).name);
fprintf('Loading Phase 3 results from: %s\n', latest_p3_results_file);
p3_data = load(latest_p3_results_file, 'bestModelInfo');
bestPipelineName = p3_data.bestModelInfo.name;
bestPipelineTestMetrics = p3_data.bestModelInfo.metrics;
fprintf('== Best pipeline identified from Phase 3: %s ==\n', bestPipelineName);

% --- Load Phase 2 Results to get the CV performance for this specific pipeline ---
p2_results_files = dir(fullfile(P.resultsPath, 'Phase2', sprintf('*_Phase2_AllPipelineResults_Strat_%s.mat', strategy)));
if isempty(p2_results_files)
    error('No Phase 2 results file found for strategy %s. Run Phase 2 first.', strategy);
end
[~,idxSortP2] = sort([p2_results_files.datenum],'descend');
latest_p2_results_file = fullfile(p2_results_files(idxSortP2(1)).folder, p2_results_files(idxSortP2(1)).name);
fprintf('Loading Phase 2 results from: %s\n', latest_p2_results_file);
p2_data = load(latest_p2_results_file, 'currentStrategyPipelinesResults', 'pipelines', 'metricNames');

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

%% 2. Generate Performance Profile Spider Plot for the Best Pipeline
fprintf('\n--- 2. Generating Performance Profile Spider Plot for %s ---\n', bestPipelineName);

% --- Prepare Data for Spider Plot ---
spider_metrics_to_plot = {'Accuracy', 'Sensitivity_WHO3', 'Specificity_WHO1', 'PPV_WHO3', 'F1_WHO3', 'AUC'};
spider_axes_labels = strrep(spider_metrics_to_plot, '_', ' ');

P_spider = zeros(2, length(spider_metrics_to_plot)); % 2 rows: CV, Test
metricNames_p2 = p2_data.metricNames;

for i = 1:length(spider_metrics_to_plot)
    metric_name = spider_metrics_to_plot{i};
    
    % Get CV metric value
    cv_metric_idx = find(strcmpi(metricNames_p2, metric_name));
    if ~isempty(cv_metric_idx)
        P_spider(1, i) = bestPipelineCVMetrics(cv_metric_idx);
    end
    
    % Get Test metric value
    if isfield(bestPipelineTestMetrics, metric_name)
        P_spider(2, i) = bestPipelineTestMetrics.(metric_name);
    end
end

% --- Create Spider Plot ---
figSpider = figure('Name', ['Performance Profile: ' bestPipelineName], 'Position', [100, 100, 700, 600]);
axesLimitsSpider = repmat([0.5; 1.0], 1, length(spider_metrics_to_plot)); % Scale from 0.5 to 1.0

spider_plot_R2019b(P_spider, ...
    'AxesLabels', spider_axes_labels, ...
    'AxesLimits', axesLimitsSpider, ...
    'AxesInterval', 5, ...
    'AxesPrecision', 2, ...
    'FillOption', 'on', ...
    'FillTransparency', [0.2, 0.1], ...
    'Color', [colorCV; colorTest], ...
    'LineWidth', 2.5, ...
    'Marker', {'o', 's'}, ...
    'MarkerSize', 80);

title({sprintf('Performance Profile: %s Pipeline', bestPipelineName); ...
       sprintf('(Outlier Strategy: %s)', strategy)}, ...
      'FontSize', 14, 'FontWeight', 'normal');
      
legend({'Mean Cross-Validation', 'Final Test Set'}, 'Location', 'southoutside', 'FontSize', 12);

spiderPlotFilenameBase = fullfile(figuresPath_output, sprintf('%s_P_Summary_SpiderPlot_%s', dateStrForFilenames, bestPipelineName));
savefig(figSpider, [spiderPlotFilenameBase, '.fig']);
exportgraphics(figSpider, [spiderPlotFilenameBase, '.tiff'], 'Resolution', 300);
fprintf('Performance Profile spider plot saved to: %s\n', [spiderPlotFilenameBase, '.tiff']);


%% 3. (Optional) Keep other summary plots as needed
% You can still include the bar charts or other summary plots from your
% original script here if you find them useful for context. For example, a
% bar chart showing the CV performance of all pipelines can still be
% valuable to justify why the "best" one was chosen.

fprintf('\n--- Visualization Script Finished ---\n');