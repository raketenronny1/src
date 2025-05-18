% run_visualize_project_results.m
%
% Script to generate key visualizations for Phase 2, 3, and 4 of the
% meningioma FT-IR classification project.
%
% Date: 2025-05-18

% run_visualize_project_results.m
%% 0. Initialization
fprintf('GENERATING PROJECT VISUALIZATIONS (Phases 2-4) - %s\n', string(datetime('now')));
clear; clc; close all;

% --- Define Paths ---
% Gehe eine Ebene vom aktuellen Pfad (...\plotting) nach oben, um zum projectRoot zu gelangen
currentScriptPath = fileparts(mfilename('fullpath')); % Pfad zum Ordner, in dem das Skript liegt
projectRoot = fileparts(fileparts(currentScriptPath)); % Geht ZWEI Ebenen hoch (von .../src/plotting/ zu .../src/ und dann zu .../)
fprintf('INFO: Project root assumed to be: %s\n', projectRoot); % Zur Überprüfung

resultsPath_main = fullfile(projectRoot, 'results');
% Stelle sicher, dass dies dein korrekter Ordnername ist:
comparisonResultsPath_P2 = fullfile(resultsPath_main, 'OutlierStrategyComparison'); % Ohne "_Results" und im korrekten "results" Ordner

% ... Rest deiner Pfaddefinitionen basierend auf dem korrigierten projectRoot ...
resultsPath_P3 = fullfile(resultsPath_main, 'Phase3');
resultsPath_P4 = fullfile(resultsPath_main, 'Phase4');
modelsPath_P3 = fullfile(projectRoot, 'models', 'Phase3');

figuresPath_output = fullfile(projectRoot, 'figures', 'ProjectSummaryFigures');
if ~isfolder(figuresPath_output), mkdir(figuresPath_output); end

% Für die Phase 2 Vergleichs-Abbildungen, die hier generiert werden
comparisonFiguresPath = fullfile(projectRoot, 'figures', 'OutlierStrategyComparison_Plots_From_VisualizeScript'); % Korrigiert hier auch den Pfad
if ~isfolder(comparisonFiguresPath), mkdir(comparisonFiguresPath); end

dateStrForFilenames = string(datetime('now','Format','yyyyMMdd'));

plottingHelpersPath = fullfile(projectRoot, 'plotting'); % Pfad zu den Helferfunktionen
 if exist(plottingHelpersPath, 'dir'), addpath(plottingHelpersPath); else, warning('Plotting helpers path not found: %s', plottingHelpersPath); end

% Plotting Defaults
plotFontSize = 10;
colorWHO1 = [0.9, 0.6, 0.4]; 
colorWHO3 = [0.4, 0.702, 0.902];
colorOR = colorWHO1; % For OR strategy in combined plots
colorAND = colorWHO3; % For AND strategy in combined plots
colorTestSet = [0.2 0.6 0.2]; % Green for test set results

%% 1. Load Phase 2 Results (Comparison of Outlier Strategies)
fprintf('\n--- 1. Loading Phase 2 Comparison Results ---\n');
overallResultsFiles_P2 = dir(fullfile(comparisonResultsPath_P2, '*_Phase2_OverallComparisonData.mat'));
if isempty(overallResultsFiles_P2)
    overallResultsFiles_P2 = dir(fullfile(resultsPath_main, '*_Phase2_OverallComparisonData.mat')); % Fallback
end
if isempty(overallResultsFiles_P2)
    error('No "*_Phase2_OverallComparisonData.mat" file found.');
end
[~,idxSort_P2] = sort([overallResultsFiles_P2.datenum],'descend');
latestOverallResultFile_P2 = fullfile(overallResultsFiles_P2(idxSort_P2(1)).folder, overallResultsFiles_P2(idxSort_P2(1)).name);
fprintf('Loading Phase 2 overall comparison results from: %s\n', latestOverallResultFile_P2);
try
    loadedData_P2 = load(latestOverallResultFile_P2, 'overallComparisonResults');
    overallComparisonResults_P2 = loadedData_P2.overallComparisonResults;
    pipelines_P2 = overallComparisonResults_P2.pipelines;
    metricNames_P2 = overallComparisonResults_P2.metricNames;
    numPipelines_P2 = length(pipelines_P2);
    pipelineNamesList_P2 = cell(numPipelines_P2, 1);
    for i=1:numPipelines_P2, pipelineNamesList_P2{i} = pipelines_P2{i}.name; end
catch ME_P2
    fprintf('ERROR loading Phase 2 overallComparisonResults: %s\n', ME_P2.message);
    rethrow(ME_P2);
end
fprintf('Phase 2 data loaded for %d pipelines.\n', numPipelines_P2);

%% 2. Generate Phase 2 Visualizations
fprintf('\n--- 2. Generating Phase 2 Visualizations ---\n');

% --- Helper function for bar charts ---
function makeBarChart(dataOR, dataAND, stdOR, stdAND, pipelineNames, metricName, titleStr, yLabelStr, figNameSuffix, outputPath, datePrefix, colors)
    fig = figure('Name', titleStr, 'Position', [100, 100, 1000, 600]);
    if isempty(dataAND) % Single strategy plot
        bar_data = dataOR;
        b = bar(bar_data);
        hold on;
        errorbar(1:length(dataOR), dataOR, stdOR, 'k.', 'HandleVisibility','off');
        b.FaceColor = colors.OR;
        legendTxt = {'T2 OR Q Strategy'};
    else % Combined plot
        bar_data = [dataOR, dataAND];
        b = bar(bar_data);
        hold on;
        numGroups = size(bar_data, 1);
        numBarsPerGroup = size(bar_data, 2);
        groupWidth = min(0.8, numBarsPerGroup/(numBarsPerGroup + 1.5));
        for iBar = 1:numBarsPerGroup
            x_centers = (1:numGroups) - groupWidth/2 + (2*iBar-1) * groupWidth / (2*numBarsPerGroup);
            if iBar == 1, errorbar(x_centers, dataOR, stdOR, 'k.', 'HandleVisibility','off');
            else, errorbar(x_centers, dataAND, stdAND, 'k.', 'HandleVisibility','off'); end
        end
        b(1).FaceColor = colors.OR;
        b(2).FaceColor = colors.AND;
        legendTxt = {'T2 OR Q Strategy', 'T2 AND Q Strategy (Consensus)'};
    end
    hold off;
    xticks(1:length(pipelineNames)); xticklabels(pipelineNames); xtickangle(45);
    ylabel(yLabelStr); title({titleStr; '(Error bars: +/-1 Std.Dev. outer CV scores)'}, 'FontWeight', 'normal');
    legend(legendTxt, 'Location', 'NorthEastOutside'); grid on;
    % Robust YLim setting
    all_means = bar_data(~isnan(bar_data));
    all_stds_or = stdOR(~isnan(stdOR) & ~isnan(dataOR));
    all_stds_and = []; if ~isempty(dataAND), all_stds_and = stdAND(~isnan(stdAND) & ~isnan(dataAND)); end
    max_vals_std = []; if ~isempty(dataOR) && ~isempty(all_stds_or), max_vals_std = [max_vals_std; dataOR(~isnan(dataOR)) + all_stds_or]; end
    if ~isempty(dataAND) && ~isempty(all_stds_and), max_vals_std = [max_vals_std; dataAND(~isnan(dataAND)) + all_stds_and]; end
    if isempty(max_vals_std), max_vals_std = all_means; end; if isempty(max_vals_std), max_vals_std = 0.1; end
    min_vals_std = []; if ~isempty(dataOR) && ~isempty(all_stds_or), min_vals_std = [min_vals_std; dataOR(~isnan(dataOR)) - all_stds_or]; end
    if ~isempty(dataAND) && ~isempty(all_stds_and), min_vals_std = [min_vals_std; dataAND(~isnan(dataAND)) - all_stds_and]; end
    if isempty(min_vals_std), min_vals_std = all_means; end; if isempty(min_vals_std), min_vals_std = 0; end
    upper_y = max(max_vals_std(:),[],'omitnan'); lower_y = min(min_vals_std(:),[],'omitnan');
    if isempty(upper_y)||isnan(upper_y),upper_y=0.1;end; if isempty(lower_y)||isnan(lower_y),lower_y=0;end
    pad = (upper_y-lower_y)*0.1; if pad==0||isnan(pad),pad=0.05;end; final_y = [max(0,lower_y-pad),upper_y+pad];
    if final_y(1)>=final_y(2),final_y=[0,max(0.1,final_y(2)+0.1)];end; ylim(final_y);
    
    filenameBase = fullfile(outputPath, sprintf('%s_BarChart_%s', datePrefix, figNameSuffix));
    exportgraphics(fig, [filenameBase, '.tiff'], 'Resolution', 300);
    savefig(fig, [filenameBase, '.fig']);
    fprintf('Bar chart "%s" saved.\n', titleStr);
    close(fig);
end

% --- Extract F2 and AUC metrics for Phase 2 bar charts ---
f2_idx_p2 = find(strcmpi(metricNames_P2, 'F2_WHO3'));
auc_idx_p2 = find(strcmpi(metricNames_P2, 'AUC'));

mean_F2_OR = NaN(numPipelines_P2,1); std_F2_OR = NaN(numPipelines_P2,1);
mean_AUC_OR = NaN(numPipelines_P2,1); std_AUC_OR = NaN(numPipelines_P2,1);
mean_F2_AND = NaN(numPipelines_P2,1); std_F2_AND = NaN(numPipelines_P2,1);
mean_AUC_AND = NaN(numPipelines_P2,1); std_AUC_AND = NaN(numPipelines_P2,1);

for p = 1:numPipelines_P2
    if isfield(overallComparisonResults_P2.Strategy_OR.allPipelinesResults{p}, 'outerFoldMetrics_mean')
        mean_F2_OR(p) = overallComparisonResults_P2.Strategy_OR.allPipelinesResults{p}.outerFoldMetrics_mean(f2_idx_p2);
        std_F2_OR(p)  = overallComparisonResults_P2.Strategy_OR.allPipelinesResults{p}.outerFoldMetrics_std(f2_idx_p2);
        mean_AUC_OR(p)= overallComparisonResults_P2.Strategy_OR.allPipelinesResults{p}.outerFoldMetrics_mean(auc_idx_p2);
        std_AUC_OR(p) = overallComparisonResults_P2.Strategy_OR.allPipelinesResults{p}.outerFoldMetrics_std(auc_idx_p2);
    end
    if isfield(overallComparisonResults_P2.Strategy_AND.allPipelinesResults{p}, 'outerFoldMetrics_mean')
        mean_F2_AND(p) = overallComparisonResults_P2.Strategy_AND.allPipelinesResults{p}.outerFoldMetrics_mean(f2_idx_p2);
        std_F2_AND(p)  = overallComparisonResults_P2.Strategy_AND.allPipelinesResults{p}.outerFoldMetrics_std(f2_idx_p2);
        mean_AUC_AND(p)= overallComparisonResults_P2.Strategy_AND.allPipelinesResults{p}.outerFoldMetrics_mean(auc_idx_p2);
        std_AUC_AND(p) = overallComparisonResults_P2.Strategy_AND.allPipelinesResults{p}.outerFoldMetrics_std(auc_idx_p2);
    end
end

% Plotting Colors for Bar Charts
barColors = struct('OR', colorOR, 'AND', colorAND);

% 1. Mean F2-WHO-3 Bar Chart for the OR Strategy with SD
makeBarChart(mean_F2_OR, [], std_F2_OR, [], pipelineNamesList_P2, 'F2_WHO3', ...
             'Mean F2-WHO3 (OR Strategy)', 'Mean F2-Score', 'P2_F2_OR_Strategy', ...
             comparisonFiguresPath, dateStrForFilenames, barColors);

% 2. Mean F2-WHO-3 Bar Chart for the AND Strategy with SD
makeBarChart(mean_F2_AND, [], std_F2_AND, [], pipelineNamesList_P2, 'F2_WHO3', ...
             'Mean F2-WHO3 (AND Strategy)', 'Mean F2-Score', 'P2_F2_AND_Strategy', ...
             comparisonFiguresPath, dateStrForFilenames, struct('OR', colorAND)); % Use AND color for single plot

% 3. Combined Mean F2-WHO-3 Bar Chart with SD for both Strategies
makeBarChart(mean_F2_OR, mean_F2_AND, std_F2_OR, std_F2_AND, pipelineNamesList_P2, 'F2_WHO3', ...
             'Combined Mean F2-WHO3 (OR vs AND Strategy)', 'Mean F2-Score', 'P2_F2_CombinedStrategies', ...
             comparisonFiguresPath, dateStrForFilenames, barColors);

% 4. Mean AUC Bar Chart for the OR Strategy with SD
makeBarChart(mean_AUC_OR, [], std_AUC_OR, [], pipelineNamesList_P2, 'AUC', ...
             'Mean AUC (OR Strategy)', 'Mean AUC', 'P2_AUC_OR_Strategy', ...
             comparisonFiguresPath, dateStrForFilenames, barColors);

% 5. Mean AUC Bar Chart for the AND Strategy with SD
makeBarChart(mean_AUC_AND, [], std_AUC_AND, [], pipelineNamesList_P2, 'AUC', ...
             'Mean AUC (AND Strategy)', 'Mean AUC', 'P2_AUC_AND_Strategy', ...
             comparisonFiguresPath, dateStrForFilenames, struct('OR', colorAND));

% 6. Combined Mean AUC Bar Chart with SD for both Strategies
makeBarChart(mean_AUC_OR, mean_AUC_AND, std_AUC_OR, std_AUC_AND, pipelineNamesList_P2, 'AUC', ...
             'Combined Mean AUC (OR vs AND Strategy)', 'Mean AUC', 'P2_AUC_CombinedStrategies', ...
             comparisonFiguresPath, dateStrForFilenames, barColors);


fprintf('\n--- Generating Revised Phase 2 Spider Plots (Pipelines as Axes) ---\n');

% --- Neuer Helfer oder angepasste Logik für Spider Plots (Pipelines auf Achsen) ---
function makePipelineAxesSpiderPlot(metricValuesOR, metricValuesAND, pipelineNames, metricNameStr, titleStrSuffix, figNameSuffix, outputPath, datePrefix, colors)
    if ~exist('spider_plot_R2019b', 'file') && ~exist('spider_plot', 'file')
        fprintf('spider_plot_R2019b.m or spider_plot.m not found. Skipping spider plot: %s\n', titleStrSuffix);
        return;
    end

    P_spider = [metricValuesOR(:)'; metricValuesAND(:)']; % Daten für Spider-Plot: 2 Zeilen (OR, AND) x N_Pipelines Spalten
    P_spider(isnan(P_spider)) = 0; % Handle NaNs für die Darstellung

    if size(P_spider, 2) ~= length(pipelineNames)
        fprintf('Mismatch between number of metric values and pipeline names for spider plot: %s. Skipping.\n', titleStrSuffix);
        return;
    end
    if isempty(P_spider)
        fprintf('No data for spider plot: %s. Skipping.\n', titleStrSuffix);
        return;
    end

    figSpider = figure('Name', sprintf('Spider Plot - %s - %s', metricNameStr, titleStrSuffix), 'Position', [200, 200, 700, 650]);
    try
        % Die Achsen des Spider-Plots sind jetzt die Pipeline-Namen
        spiderAxesLabels = pipelineNames;
        
        % Achsen-Limits für Spider-Plot (z.B. 0 bis 1 für Metriken wie AUC, F2)
        % Du kannst dies dynamischer gestalten oder fix lassen.
        minValue = 0; % min(P_spider(:),[],'omitnan');
        maxValue = 1; % max(P_spider(:),[],'omitnan');
        if minValue >= maxValue, maxValue = minValue + 0.1; end % Fallback
        if isnan(minValue) || isnan(maxValue)
            minValue = 0; maxValue = 1;
        end

        % Sicherstellen, dass die Limits mindestens einen kleinen Bereich abdecken
        if maxValue - minValue < 0.1 
            maxValue = minValue + 0.1;
        end
        % Für Metriken wie F2 und AUC ist ein Bereich von 0-1 üblich
        commonSpiderAxesLimits = repmat([0; 1], 1, length(spiderAxesLabels)); 


        if exist('spider_plot_R2019b', 'file')
            spider_plot_R2019b(P_spider, ...
                'AxesLabels', spiderAxesLabels, ...
                'AxesLimits', commonSpiderAxesLimits, ...
                'FillOption', {'on', 'on'}, ...
                'FillTransparency', [0.15, 0.15], ...
                'Color', [colors.OR; colors.AND], ...
                'LineWidth', 2, ...
                'Marker', {'o', 's'}, ...
                'MarkerSize', 60, ...
                'AxesFontSize', 9, ...
                'LabelFontSize', 10);
        elseif exist('spider_plot', 'file') % Fallback zur älteren Version
             spider_plot(P_spider, ...
                'AxesLabels', spiderAxesLabels, ...
                'AxesLimits', commonSpiderAxesLimits, ...
                'FillOption', 'on', ...
                'FillTransparency', [0.15, 0.15], ...
                'Color', [colors.OR; colors.AND], ...
                'LineWidth', 2, ...
                'Marker', 'o', ... % Ältere Version hat ggf. andere Marker-Optionen
                'MarkerSize', 8); % Ältere Version hat ggf. andere Marker-Optionen
        end

        title(sprintf('%s Performance Comparison: %s', metricNameStr, titleStrSuffix), 'FontWeight', 'normal', 'FontSize', plotFontSize);
        legend({'T2 OR Q Strategy', 'T2 AND Q Strategy (Consensus)'}, 'Location', 'southoutside', 'Orientation','horizontal', 'FontSize', plotFontSize-1);
        
        filenameBase = fullfile(outputPath, sprintf('%s_Spider_%s_%s', datePrefix, figNameSuffix, metricNameStr));
        exportgraphics(figSpider, [filenameBase, '.tiff'], 'Resolution', 300);
        savefig(figSpider, [filenameBase, '.fig']);
        fprintf('Pipeline-Axes Spider plot for "%s - %s" saved.\n', metricNameStr, titleStrSuffix);
    catch ME_spider
        fprintf('Error generating pipeline-axes spider plot for %s: %s\n', titleStrSuffix, ME_spider.message);
        disp(ME_spider.getReport);
    end
    if isgraphics(figSpider, 'figure'), close(figSpider); end
end

% --- Erzeuge die gewünschten Spider Plots ---

% 1. Spider Plot für mittlere AUC-Werte
% pipelineNamesList_P2 sind die Achsenbeschriftungen
% mean_AUC_OR und mean_AUC_AND sind die Datenreihen
makePipelineAxesSpiderPlot(mean_AUC_OR, mean_AUC_AND, pipelineNamesList_P2, 'Mean AUC', ...
                           'Pipeline Comparison', 'P2_AUC_Pipelines', ...
                           comparisonFiguresPath, dateStrForFilenames, barColors);

% 2. Spider Plot für mittlere F2-WHO3-Werte
makePipelineAxesSpiderPlot(mean_F2_OR, mean_F2_AND, pipelineNamesList_P2, 'Mean F2-WHO3', ...
                           'Pipeline Comparison', 'P2_F2_Pipelines', ...
                           comparisonFiguresPath, dateStrForFilenames, barColors);

% (Die alte Sektion mit den Spider-Plots, die Metriken auf den Achsen hatten, kann entfernt/auskommentiert werden)
% % --- Data for Spider Plots (alte Version, die Metriken auf Achsen hatte) ---
% % metricsForF2Spider = {'F2_WHO3', 'Sensitivity_WHO3', 'Specificity_WHO1', 'PPV_WHO3', 'NPV_WHO1'};
% % metricsForAUCSpider = {'AUC', 'Accuracy', 'Sensitivity_WHO3', 'Specificity_WHO1'};
% % ... (alte dataF2_OR, dataF2_AND, dataAUC_OR, dataAUC_AND Extraktion) ...
% 
% % 7. Spider Plot for F2-related metrics (alte Version)
% % makeSpiderPlot(dataF2_OR, dataF2_AND, pipelineNamesList_P2, metricsForF2Spider, ...
% %                'F2-Focused Performance Metrics', 'P2_F2_Metrics', ...
% %                comparisonFiguresPath, dateStrForFilenames, barColors);
% 
% % 8. Spider Plot for AUC-related metrics (alte Version)
% % makeSpiderPlot(dataAUC_OR, dataAUC_AND, pipelineNamesList_P2, metricsForAUCSpider, ...
% %                'AUC-Focused Performance Metrics', 'P2_AUC_Metrics', ...
% %                comparisonFiguresPath, dateStrForFilenames, barColors);



%% 3. Load Phase 3 Results (Final Model Evaluation)
fprintf('\n--- 3. Loading Phase 3 Results ---\n');
% Determine best pipeline overall (e.g., from AND strategy if it was better, or a specific one)
% For this visualization script, we might need to assume a "best" model was chosen
% or visualize for a specific pipeline, e.g., MRMRLDA if it was the Phase 3 choice.
finalModelFiles = dir(fullfile(modelsPath_P3, '*_Phase3_FinalMRMRLDA_Model.mat')); % Assuming MRMRLDA was chosen
if isempty(finalModelFiles)
    fprintf('No final Phase 3 model file found. Skipping Phase 3 visualizations.\n');
    P3_data_loaded = false;
else
    [~,idxSort_P3M] = sort([finalModelFiles.datenum],'descend');
    latestFinalModelFile = fullfile(modelsPath_P3, finalModelFiles(idxSort_P3M(1)).name);
    fprintf('Loading final model package from: %s\n', latestFinalModelFile);
    try
        load(latestFinalModelFile, 'finalModelPackage');
        P3_test_performance_spectrum = finalModelPackage.testSetPerformance; % Spectrum-level
        P3_data_loaded = true;
    catch ME_P3M
        fprintf('ERROR loading final model package: %s. Skipping Phase 3 spectrum-level viz.\n', ME_P3M.message);
        P3_data_loaded = false;
    end
end

% Load probe-level results for Phase 3
probeResultsFiles_P3 = dir(fullfile(resultsPath_P3, '*_Phase3_ProbeLevelTestSetResults.mat'));
if isempty(probeResultsFiles_P3)
    fprintf('No Phase 3 probe-level results file found. Skipping Phase 3 probe-level visualizations.\n');
    P3_probe_data_loaded = false;
else
    [~,idxSort_P3P] = sort([probeResultsFiles_P3.datenum],'descend');
    latestProbeResultFile = fullfile(resultsPath_P3, probeResultsFiles_P3(idxSort_P3P(1)).name);
    fprintf('Loading Phase 3 probe-level results from: %s\n', latestProbeResultFile);
    try
        load(latestProbeResultFile, 'probeLevelResults', 'probeLevelPerfMetrics');
        P3_probe_performance = probeLevelPerfMetrics;
        P3_dataTable_probes_with_scores = probeLevelResults; % Contains Diss_ID, True_WHO_Grade_Numeric, Mean_WHO3_Probability etc.
        P3_probe_data_loaded = true;
    catch ME_P3P
         fprintf('ERROR loading Phase 3 probe-level results: %s. Skipping Phase 3 probe-level viz.\n', ME_P3P.message);
        P3_probe_data_loaded = false;
    end
end


%% 4. Generate Phase 3 Visualizations
fprintf('\n--- 4. Generating Phase 3 Visualizations ---\n');
if P3_data_loaded
    % --- Confusion Matrix for Test Set (Spectrum-Level) ---
    % This requires y_true_test and y_pred_test for the final model.
    % These are not directly in finalModelPackage.testSetPerformance.
    % We'd need to re-predict or load them if saved separately in Phase 3.
    % For now, let's plot the performance metrics table from P3_test_performance_spectrum.
    
    figP3SpecMetrics = figure('Name', 'Phase 3 Final Model - Test Set Performance (Spectrum-Level)');
    uitable(figP3SpecMetrics, 'Data', struct2cell(P3_test_performance_spectrum)', ...
            'ColumnName', fieldnames(P3_test_performance_spectrum), ...
            'RowName', {'Value'}, 'Units', 'Normalized', 'Position', [0.1 0.1 0.8 0.8]);
    title('Final Model Test Set Performance (Spectrum-Level)');
    metricsTableP3SpecFilename = fullfile(figuresPath_output, sprintf('%s_P3_TestSetPerformance_SpectrumLevel_Table.tiff', dateStrForFilenames));
    exportgraphics(figP3SpecMetrics, metricsTableP3SpecFilename, 'Resolution', 150);
    fprintf('Phase 3 Spectrum-Level Performance Table saved to: %s\n', metricsTableP3SpecFilename);
    close(figP3SpecMetrics);
end

if P3_probe_data_loaded
    % --- Confusion Matrix for Test Set (Probe-Level) ---
    % From run_phase3_final_evaluation.m, you plotted cm_probe. We can recreate this.
    y_true_probe = P3_dataTable_probes_with_scores.True_WHO_Grade_Numeric;
    y_pred_probe_mean_prob = P3_dataTable_probes_with_scores.Predicted_WHO_Grade_Numeric_MeanProb; % From MeanProb>0.5
    
    if ~isempty(y_true_probe) && ~isempty(y_pred_probe_mean_prob) && length(unique(y_true_probe(~isnan(y_true_probe)))) > 1
        figP3ConfMatProbe = figure('Name', 'Phase 3 - Confusion Matrix (Probe-Level)');
        cm_probe = confusionchart(y_true_probe, y_pred_probe_mean_prob, ...
            'ColumnSummary','column-normalized', 'RowSummary','row-normalized', ...
            'Title', sprintf('Probe-Level Confusion Matrix (Mean Prob > 0.5, F2: %.3f)', P3_probe_performance.F2_WHO3));
        confMatP3ProbeFilename = fullfile(figuresPath_output, sprintf('%s_P3_ConfusionMatrix_ProbeLevel.tiff', dateStrForFilenames));
        exportgraphics(figP3ConfMatProbe, confMatP3ProbeFilename, 'Resolution', 300);
        savefig(figP3ConfMatProbe, strrep(confMatP3ProbeFilename, '.tiff','.fig'));
        fprintf('Phase 3 Probe-Level Confusion Matrix saved to: %s\n', confMatP3ProbeFilename);
        close(figP3ConfMatProbe);
    else
        fprintf('Skipping Phase 3 Probe-Level Confusion Matrix: Not enough data or classes.\n');
    end

    % --- Probe-Level Probability Distribution Plot ---
    % (From run_phase3_final_evaluation.m, Section 5)
    figP3ProbDist = figure('Name', 'Phase 3 - Probe-Level Mean WHO-3 Probabilities (Test Set)', 'Position', [100, 100, 900, 700]);
    hold on; jitterAmount = 0.02; 
    probes_true_who1 = P3_dataTable_probes_with_scores(P3_dataTable_probes_with_scores.True_WHO_Grade_Numeric == 1, :);
    probes_true_who3 = P3_dataTable_probes_with_scores(P3_dataTable_probes_with_scores.True_WHO_Grade_Numeric == 3, :);
    h_p3_1 = []; h_p3_3 = [];
    if ~isempty(probes_true_who1)
        x_coords_who1 = 1 + (rand(height(probes_true_who1),1) - 0.5) * jitterAmount * 2;
        h_p3_1 = scatter(x_coords_who1, probes_true_who1.Mean_WHO3_Probability, 70, 'o', 'MarkerEdgeColor','k','MarkerFaceColor',colorWHO1,'LineWidth',1,'DisplayName','True WHO-1 Probes');
    end
    if ~isempty(probes_true_who3)
        x_coords_who3 = 2 + (rand(height(probes_true_who3),1) - 0.5) * jitterAmount * 2;
        h_p3_3 = scatter(x_coords_who3, probes_true_who3.Mean_WHO3_Probability, 70, 's', 'MarkerEdgeColor','k','MarkerFaceColor',colorWHO3,'LineWidth',1,'DisplayName','True WHO-3 Probes');
    end
    plot([0.5 2.5], [0.5 0.5], 'k--', 'DisplayName', 'Decision Threshold (0.5)');
    % Add text labels (optional, can get crowded)
    % text(x_coords_all + 0.02, y_coords_all, labels_all, 'FontSize', 8);
    hold off; xticks([1 2]); xticklabels({'True WHO-1', 'True WHO-3'}); xlim([0.5 2.5]); ylim([0 1]);
    ylabel('Mean Predicted Probability of WHO-3'); title('Probe-Level Classification Probabilities (Test Set)'); grid on;
    if ~isempty(h_p3_1) || ~isempty(h_p3_3), legend([h_p3_1, h_p3_3], 'Location', 'best'); end; set(gca, 'FontSize', 12);
    probDistP3Filename = fullfile(figuresPath_output, sprintf('%s_P3_ProbeLevelProbabilities.tiff', dateStrForFilenames));
    exportgraphics(figP3ProbDist, probDistP3Filename, 'Resolution', 300);
    savefig(figP3ProbDist, strrep(probDistP3Filename, '.tiff','.fig'));
    fprintf('Phase 3 Probe-Level Probability Distribution saved to: %s\n', probDistP3Filename);
    close(figP3ProbDist);
end

%% 5. Load Phase 4 Results (Feature Interpretation)
fprintf('\n--- 5. Loading Phase 4 Results ---\n');
featureTableFile = fullfile(resultsPath_P4, sprintf('%s_Phase4_FeatureImportanceTable_WithPvalues.csv', dateStrForFilenames)); % Use latest or specific
altFeatureTableFiles = dir(fullfile(resultsPath_P4, '*_Phase4_FeatureImportanceTable_WithPvalues.csv'));
if ~exist(featureTableFile, 'file') && ~isempty(altFeatureTableFiles)
    [~,idxSort_P4F] = sort([altFeatureTableFiles.datenum],'descend');
    featureTableFile = fullfile(resultsPath_P4, altFeatureTableFiles(idxSort_P4F(1)).name);
    fprintf('Using latest feature importance table: %s\n', featureTableFile);
end

if exist(featureTableFile, 'file')
    try
        P4_feature_importance_table = readtable(featureTableFile);
        P4_data_loaded = true;
        fprintf('Phase 4 Feature Importance Table loaded from: %s\n', featureTableFile);
    catch ME_P4F
        fprintf('ERROR loading Phase 4 feature importance table: %s\n', ME_P4F.message);
        P4_data_loaded = false;
    end
else
    fprintf('Phase 4 feature importance table not found. Skipping Phase 4 visualizations.\n');
    P4_data_loaded = false;
end

%% 6. Generate Phase 4 Visualizations
fprintf('\n--- 6. Generating Phase 4 Visualizations ---\n');
if P4_data_loaded && P3_data_loaded % Need P3 model package for some P4 plots
    % --- LDA Coefficient Spectrum ---
    % (From run_phase4_feature_interpretation.m, Section 3)
    figP4Coeff = figure('Name', 'Phase 4 - LDA Coefficient Spectrum', 'Position', [100, 100, 900, 600]);
    if ismember('BinnedWavenumber_cm_neg1', P4_feature_importance_table.Properties.VariableNames) && ...
       ismember('LDACoefficient', P4_feature_importance_table.Properties.VariableNames)
        
        plot_wavenumbers_p4 = P4_feature_importance_table.BinnedWavenumber_cm_neg1;
        plot_coeffs_p4 = P4_feature_importance_table.LDACoefficient;
        [plot_wavenumbers_p4_sorted, sortIdx_p4] = sort(plot_wavenumbers_p4);
        plot_coeffs_p4_sorted = plot_coeffs_p4(sortIdx_p4);

        stem(plot_wavenumbers_p4_sorted, plot_coeffs_p4_sorted, 'filled', 'MarkerSize', 4);
        hold on; plot(plot_wavenumbers_p4_sorted, zeros(size(plot_wavenumbers_p4_sorted)), 'k--'); hold off;
        xlabel(sprintf('Gebinnte Wellenzahl (cm^{-1}) - Binning Faktor %d', finalModelPackage.binningFactor));
        ylabel('LDA Koeffizientenwert');
        title({'LDA Koeffizienten der MRMR-selektierten Merkmale (WHO-1 vs WHO-3)'; 'Positive Werte sprechen für WHO-3'});
        grid on; ax_p4c = gca; ax_p4c.XDir = 'reverse'; 
        if ~isempty(plot_wavenumbers_p4_sorted), xlim([min(plot_wavenumbers_p4_sorted)-5 max(plot_wavenumbers_p4_sorted)+5]); else xlim(plotXLim); end

        ldaCoeffPlotP4Filename = fullfile(figuresPath_output, sprintf('%s_P4_LDACoeffSpectrum.tiff', dateStrForFilenames));
        exportgraphics(figP4Coeff, ldaCoeffPlotP4Filename, 'Resolution', 300);
        savefig(figP4Coeff, strrep(ldaCoeffPlotP4Filename, '.tiff','.fig'));
        fprintf('Phase 4 LDA Coefficient Spectrum saved to: %s\n', ldaCoeffPlotP4Filename);
    else
        fprintf('Skipping LDA Coefficient plot: Required columns not in feature table.\n');
    end
    if isgraphics(figP4Coeff), close(figP4Coeff); end

    % --- Mean Spectra with Highlighted Features ---
    % (From run_phase4_feature_interpretation.m, Section 4)
    % This requires loading training data again, or having saved mean spectra.
    % For simplicity, assuming finalModelPackage has what's needed (selectedWavenumbers, binningFactor)
    % and the mean spectra plot from Phase 4 script could be adapted or called.
    % If you have a separate function for Plot 4 from phase4 script, call it here.
    % Otherwise, this part would re-implement that plotting logic.
    fprintf('Phase 4 Mean Spectra with Highlighted Features: Code from run_phase4_feature_interpretation.m Section 4 would go here.\n');
    fprintf('This typically requires X_train_full and y_train_full to recalculate means, or loading them if saved.\n');
    % If you have finalModelPackage.selectedWavenumbers and finalModelPackage.LDAModel.Coeffs,
    % and can load the training data used for run_phase4.m, you can replicate the plot.
    % Example (conceptual):
    % if exist('finalModelPackage','var') && isfield(finalModelPackage, 'trainingDataFile')
    %    load(finalModelPackage.trainingDataFile, 'X_train_no_outliers', 'y_train_numeric_no_outliers');
    %    % ... then the plotting logic from run_phase4_feature_interpretation.m Section 4 ...
    %    fprintf('Replicated mean spectra plot from phase 4 here and saved it.\n');
    % else
    %    fprintf('Skipping mean spectra highlight plot - missing data/model info.\n');
    % end


end


%% 7. Additional Suggested Visualizations
fprintf('\n--- 7. Suggestions for Additional Visualizations ---\n');

% --- ROC Curves for Best Pipelines (from Phase 2, per strategy) ---
fprintf('Suggestion: Plot ROC curves from outer CV folds for best OR and best AND pipelines.\n');
% To do this:
% 1. Identify best pipeline for OR and AND strategy (from overallComparisonResults_P2).
% 2. The `allPipelinesResults{iPipeline}.outerFoldMetrics_raw` should ideally store
%    y_true and y_scores for each outer fold if `perform_inner_cv` or the outer loop saves them.
%    If not, `run_phase2_model_selection_comparative.m` needs modification to save these.
%    Let's assume you could get `all_y_true_folds_OR`, `all_y_scores_folds_OR` for best OR pipeline,
%    and similarly for AND.
%
% Example (conceptual, if data is available):
% figROC_P2 = figure; tiledlayout(1,2);
% axROC_OR = nexttile; title(axROC_OR, 'Best OR Pipeline - Outer Fold ROCs'); hold(axROC_OR, 'on');
% % for each outer fold of best OR pipeline:
% %   [Xroc,Yroc,~,AUCroc] = perfcurve(y_true_fold, y_scores_fold_positive_class, positive_label);
% %   plot(axROC_OR, Xroc, Yroc, 'DisplayName', sprintf('Fold %d (AUC=%.2f)', k, AUCroc));
% hold(axROC_OR, 'off'); xlabel(axROC_OR,'FPR'); ylabel(axROC_OR,'TPR'); grid(axROC_OR,'on');
%
% axROC_AND = nexttile; title(axROC_AND, 'Best AND Pipeline - Outer Fold ROCs');  hold(axROC_AND, 'on');
% % for each outer fold of best AND pipeline:
% %   plot(axROC_AND, Xroc, Yroc, 'DisplayName', sprintf('Fold %d (AUC=%.2f)', k, AUCroc));
% hold(axROC_AND, 'off'); xlabel(axROC_AND,'FPR'); ylabel(axROC_AND,'TPR'); grid(axROC_AND,'on');
% sgtitle('Phase 2: ROC Curves for Best Pipelines (Outer CV Folds)');
% save/export figROC_P2...

% --- Box Plots of Performance Metrics (Phase 2) ---
fprintf('Suggestion: Box plots showing distribution of a key metric (e.g., F2) across outer folds for each pipeline & strategy.\n');
% For each pipeline and each strategy, you have outerFoldMetrics_raw.
% figBox_P2 = figure;
% f2_scores_OR_all_pipelines = []; % Collect all F2 scores for OR
% f2_scores_AND_all_pipelines = []; % Collect all F2 scores for AND
% pipelineGroupIdx_OR = []; pipelineGroupIdx_AND = [];
% for p = 1:numPipelines_P2
%    f2_or = overallComparisonResults_P2.Strategy_OR.allPipelinesResults{p}.outerFoldMetrics_raw(:, f2_idx_p2);
%    f2_scores_OR_all_pipelines = [f2_scores_OR_all_pipelines; f2_or(~isnan(f2_or))];
%    pipelineGroupIdx_OR = [pipelineGroupIdx_OR; repmat(p, sum(~isnan(f2_or)), 1)];
%    % Similarly for AND
% end
% subplot(1,2,1); boxplot(f2_scores_OR_all_pipelines, pipelineGroupIdx_OR); title('F2 Scores (OR Strategy)'); xticklabels(pipelineNamesList_P2); xtickangle(45);
% subplot(1,2,2); % For AND
% save/export figBox_P2...

% --- Test Set ROC Curve (Phase 3) ---
fprintf('Suggestion: Plot ROC curve for the final model on the test set (spectrum-level).\n');
% This needs y_true_test and y_scores_test (for positive class) for the final model.
% finalModelPackage from Phase 3 run_phase3_final_evaluation.m often has y_pred_test and y_scores_test.
% If finalModelPackage has `y_test_full_numeric` and `scores_for_positive_class_test`:
% if P3_data_loaded && isfield(finalModelPackage, 'y_test_full_numeric_from_eval') % Hypothetical field name
%    [Xroc_test,Yroc_test,~,AUC_test] = perfcurve(finalModelPackage.y_test_full_numeric_from_eval, ...
%                                                 finalModelPackage.scores_for_positive_class_test_from_eval, 3);
%    figure; plot(Xroc_test, Yroc_test); xlabel('FPR'); ylabel('TPR');
%    title(sprintf('Test Set ROC Curve (AUC = %.3f)', AUC_test)); grid on;
%    % save/export ...
% end

fprintf('\n--- All Requested and Suggested Visualizations Attempted ---\n');