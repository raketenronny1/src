% run_visualize_project_results.m
%
% Script to generate key visualizations for Phase 2, 3, and 4 of the
% meningioma FT-IR classification project.
%
% Date: 2025-05-18 (Spider plot styling: no main fill, increased tile spacing, label offset)

%% 0. Initialization
fprintf('GENERATING PROJECT VISUALIZATIONS (Phases 2-4) - %s\n', string(datetime('now')));
clear; clc; close all;

% --- Define Paths ---
currentScriptPath = fileparts(mfilename('fullpath')); 
projectRoot = fileparts(fileparts(currentScriptPath)); 
fprintf('INFO: Project root assumed to be: %s\n', projectRoot);

resultsPath_main = fullfile(projectRoot, 'results');
comparisonResultsPath_P2 = fullfile(resultsPath_main, 'OutlierStrategyComparison'); 

resultsPath_P3 = fullfile(resultsPath_main, 'Phase3');
resultsPath_P4 = fullfile(resultsPath_main, 'Phase4');
modelsPath_P3 = fullfile(projectRoot, 'models', 'Phase3');

figuresPath_output = fullfile(projectRoot, 'figures', 'ProjectSummaryFigures'); 
if ~isfolder(figuresPath_output), mkdir(figuresPath_output); end

comparisonFiguresPath = fullfile(projectRoot, 'figures', 'OutlierStrategyComparison_Plots_From_VisualizeScript');
if ~isfolder(comparisonFiguresPath), mkdir(comparisonFiguresPath); end

dateStrForFilenames = string(datetime('now','Format','yyyyMMdd'));

plottingHelpersPath = fullfile(projectRoot, 'plotting'); 
if exist(plottingHelpersPath, 'dir')
    addpath(plottingHelpersPath); 
    fprintf('Added to path: %s\n', plottingHelpersPath);
else
    warning('Plotting helpers path not found: %s. Ensure spider_plot_R2019b.m is on the MATLAB path or in this directory.', plottingHelpersPath);
end

% Plotting Defaults
plotFontSize = 10; 
colorStrategyOR = [0.5, 0, 0.5];    % Purple for OR line
colorStrategyAND = [0, 0.5, 0.5];   % Teal for AND line
colorStdDevShadeOR = [0.8, 0.65, 0.8]; % Lighter Purple for OR shade
colorStdDevShadeAND = [0.6, 0.8, 0.8]; % Lighter Teal for AND shade
colorWHO1 = [0.9, 0.6, 0.4]; 
colorWHO3 = [0.4, 0.702, 0.902]; 

colorTestSet = [0.2 0.6 0.2]; 
P2_data_loaded_successfully = false; 

%% 1. Load Phase 2 Results (Comparison of Outlier Strategies)
% ... (This section remains the same as your working version) ...
fprintf('\n--- 1. Loading Phase 2 Comparison Results ---\n');
overallResultsFiles_P2 = dir(fullfile(comparisonResultsPath_P2, '*_Phase2_OverallComparisonData.mat'));
if isempty(overallResultsFiles_P2)
    overallResultsFiles_P2 = dir(fullfile(resultsPath_main, '*_Phase2_OverallComparisonData.mat')); 
end

if isempty(overallResultsFiles_P2)
    error('No "*_Phase2_OverallComparisonData.mat" file found in %s or %s.', comparisonResultsPath_P2, resultsPath_main);
end

[~,idxSort_P2] = sort([overallResultsFiles_P2.datenum],'descend');
latestOverallResultFile_P2 = fullfile(overallResultsFiles_P2(idxSort_P2(1)).folder, overallResultsFiles_P2(idxSort_P2(1)).name);
fprintf('Loading Phase 2 overall comparison results from: %s\n', latestOverallResultFile_P2);

try
    loadedData_P2 = load(latestOverallResultFile_P2, 'overallComparisonResults');
    overallComparisonResults_P2 = loadedData_P2.overallComparisonResults;
    
    if isfield(overallComparisonResults_P2, 'Strategy_OR') && ...
       isfield(overallComparisonResults_P2.Strategy_OR, 'allPipelinesResults') && ...
       isfield(overallComparisonResults_P2, 'Strategy_AND') && ...
       isfield(overallComparisonResults_P2.Strategy_AND, 'allPipelinesResults') && ...
       isfield(overallComparisonResults_P2, 'pipelines') && ...
       isfield(overallComparisonResults_P2, 'metricNames')

        pipelines_P2 = overallComparisonResults_P2.pipelines;
        metricNames_P2 = overallComparisonResults_P2.metricNames;
        numPipelines_P2 = length(pipelines_P2);
        pipelineNamesList_P2 = cell(numPipelines_P2, 1);
        for i=1:numPipelines_P2, pipelineNamesList_P2{i} = pipelines_P2{i}.name; end
        
        P2_data_loaded_successfully = true; 
        fprintf('Phase 2 data loaded successfully for %d pipelines.\n', numPipelines_P2);
    else
        P2_data_loaded_successfully = false; 
        error('Phase 2 overallComparisonResults not loaded completely or in expected format from: %s', latestOverallResultFile_P2);
    end
catch ME_P2
    P2_data_loaded_successfully = false; 
    fprintf('ERROR loading Phase 2 overallComparisonResults: %s\n', ME_P2.message);
    rethrow(ME_P2);
end


%% 2. Generate Phase 2 Visualizations
fprintf('\n--- 2. Generating Phase 2 Visualizations ---\n');

if P2_data_loaded_successfully
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
    
    strategyBarColors = struct('OR', colorStrategyOR, 'AND', colorStrategyAND);

    makeBarChart(mean_F2_OR, [], std_F2_OR, [], pipelineNamesList_P2, 'F2_WHO3', ...
                 'Mean F2-WHO3 (OR Strategy)', 'Mean F2-Score', 'P2_F2_OR_Strategy', ...
                 comparisonFiguresPath, dateStrForFilenames, strategyBarColors);
    makeBarChart(mean_F2_AND, [], std_F2_AND, [], pipelineNamesList_P2, 'F2_WHO3', ...
                 'Mean F2-WHO3 (AND Strategy)', 'Mean F2-Score', 'P2_F2_AND_Strategy', ...
                 comparisonFiguresPath, dateStrForFilenames, struct('OR', colorStrategyAND)); 
    makeBarChart(mean_F2_OR, mean_F2_AND, std_F2_OR, std_F2_AND, pipelineNamesList_P2, 'F2_WHO3', ...
                 'Combined Mean F2-WHO3 (OR vs AND Strategy)', 'Mean F2-Score', 'P2_F2_CombinedStrategies', ...
                 comparisonFiguresPath, dateStrForFilenames, strategyBarColors);
    makeBarChart(mean_AUC_OR, [], std_AUC_OR, [], pipelineNamesList_P2, 'AUC', ...
                 'Mean AUC (OR Strategy)', 'Mean AUC', 'P2_AUC_OR_Strategy', ...
                 comparisonFiguresPath, dateStrForFilenames, strategyBarColors);
    makeBarChart(mean_AUC_AND, [], std_AUC_AND, [], pipelineNamesList_P2, 'AUC', ...
                 'Mean AUC (AND Strategy)', 'Mean AUC', 'P2_AUC_AND_Strategy', ...
                 comparisonFiguresPath, dateStrForFilenames, struct('OR', colorStrategyAND));
    makeBarChart(mean_AUC_OR, mean_AUC_AND, std_AUC_OR, std_AUC_AND, pipelineNamesList_P2, 'AUC', ...
                 'Combined Mean AUC (OR vs AND Strategy)', 'Mean AUC', 'P2_AUC_CombinedStrategies', ...
                 comparisonFiguresPath, dateStrForFilenames, strategyBarColors);
else
    fprintf('Skipping Phase 2 Bar Chart generation as P2_data_loaded_successfully is false.\n');
end

% --- Tiled Spider Plots ---
if P2_data_loaded_successfully && ...
   exist('mean_AUC_OR', 'var') && exist('mean_F2_OR', 'var') && ... 
   ~isempty(mean_AUC_OR) && ~isempty(mean_F2_OR) 

    spiderPlotLineColors = struct('OR', colorStrategyOR, 'AND', colorStrategyAND);
    spiderPlotShadeColors = struct('OR', colorStdDevShadeOR, 'AND', colorStdDevShadeAND);

    % AUC Spider Plot (0.6 to 1.0)
    aucSpiderAxesLimits = repmat([0.6; 1.0], 1, numPipelines_P2);
    aucSpiderAxesInterval = 4; 
    makeTiledSpiderPlot(mean_AUC_OR, std_AUC_OR, mean_AUC_AND, std_AUC_AND, ...
                        pipelineNamesList_P2, 'Mean AUC', aucSpiderAxesLimits, aucSpiderAxesInterval, ...
                        'AUC_Pipelines', comparisonFiguresPath, dateStrForFilenames, spiderPlotLineColors, spiderPlotShadeColors, plotFontSize);

    % F2-WHO3 Spider Plot (0.6 to 1.0)
    f2SpiderAxesLimits = repmat([0.6; 1.0], 1, numPipelines_P2);
    f2SpiderAxesInterval = 4; 
    makeTiledSpiderPlot(mean_F2_OR, std_F2_OR, mean_F2_AND, std_F2_AND, ...
                        pipelineNamesList_P2, 'Mean F2-WHO3', f2SpiderAxesLimits, f2SpiderAxesInterval, ...
                        'F2_Pipelines', comparisonFiguresPath, dateStrForFilenames, spiderPlotLineColors, spiderPlotShadeColors, plotFontSize);
else
    fprintf('Data for Tiled Spider Plots not fully available. Skipping these plots.\n');
    if ~(exist('P2_data_loaded_successfully', 'var') && P2_data_loaded_successfully)
         fprintf('Reason: Phase 2 data was not successfully loaded or initialized.\n');
    else
         fprintf('Reason: One or more required metric arrays (mean_AUC_OR, mean_F2_OR etc.) are missing or empty.\n');
    end
end


%% 3. Load Phase 3 Results 
% ... (This section remains unchanged) ...
fprintf('\n--- 3. Loading Phase 3 Results ---\n');
P3_data_loaded = false; 
P3_probe_data_loaded = false; 
finalModelPackage = []; 

finalModelFiles = dir(fullfile(modelsPath_P3, '*_Phase3_FinalMRMRLDA_Model.mat'));
if isempty(finalModelFiles)
    fprintf('No final Phase 3 model file found. Skipping Phase 3 spectrum-level visualizations.\n');
else
    [~,idxSort_P3M] = sort([finalModelFiles.datenum],'descend');
    latestFinalModelFile = fullfile(modelsPath_P3, finalModelFiles(idxSort_P3M(1)).name);
    fprintf('Loading final model package from: %s\n', latestFinalModelFile);
    try
        loaded_P3_model_package = load(latestFinalModelFile, 'finalModelPackage');
        finalModelPackage = loaded_P3_model_package.finalModelPackage; 
        P3_test_performance_spectrum = finalModelPackage.testSetPerformance; 
        P3_data_loaded = true;
    catch ME_P3M
        fprintf('ERROR loading final model package: %s. Skipping Phase 3 spectrum-level viz.\n', ME_P3M.message);
    end
end

probeResultsFiles_P3 = dir(fullfile(resultsPath_P3, '*_Phase3_ProbeLevelTestSetResults.mat'));
if isempty(probeResultsFiles_P3)
    fprintf('No Phase 3 probe-level results file found. Skipping Phase 3 probe-level visualizations.\n');
else
    [~,idxSort_P3P] = sort([probeResultsFiles_P3.datenum],'descend');
    latestProbeResultFile = fullfile(resultsPath_P3, probeResultsFiles_P3(idxSort_P3P(1)).name);
    fprintf('Loading Phase 3 probe-level results from: %s\n', latestProbeResultFile);
    try
        load(latestProbeResultFile, 'probeLevelResults', 'probeLevelPerfMetrics');
        P3_probe_performance = probeLevelPerfMetrics;
        P3_dataTable_probes_with_scores = probeLevelResults; 
        P3_probe_data_loaded = true;
    catch ME_P3P
         fprintf('ERROR loading Phase 3 probe-level results: %s. Skipping Phase 3 probe-level viz.\n', ME_P3P.message);
    end
end

%% 4. Generate Phase 3 Visualizations
% ... (This section remains unchanged) ...
fprintf('\n--- 4. Generating Phase 3 Visualizations ---\n');
if P3_data_loaded
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
    y_true_probe = P3_dataTable_probes_with_scores.True_WHO_Grade_Numeric;
    y_pred_probe_mean_prob = P3_dataTable_probes_with_scores.Predicted_WHO_Grade_Numeric_MeanProb; 
    
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
    hold off; xticks([1 2]); xticklabels({'True WHO-1', 'True WHO-3'}); xlim([0.5 2.5]); ylim([0 1]);
    ylabel('Mean Predicted Probability of WHO-3'); title('Probe-Level Classification Probabilities (Test Set)'); grid on;
    if ~isempty(h_p3_1) || ~isempty(h_p3_3), legend([h_p3_1, h_p3_3], 'Location', 'best'); end; set(gca, 'FontSize', 12);
    probDistP3Filename = fullfile(figuresPath_output, sprintf('%s_P3_ProbeLevelProbabilities.tiff', dateStrForFilenames));
    exportgraphics(figP3ProbDist, probDistP3Filename, 'Resolution', 300);
    savefig(figP3ProbDist, strrep(probDistP3Filename, '.tiff','.fig'));
    fprintf('Phase 3 Probe-Level Probability Distribution saved to: %s\n', probDistP3Filename);
    close(figP3ProbDist);
end

%% 5. Load Phase 4 Results
% ... (This section remains unchanged) ...
fprintf('\n--- 5. Loading Phase 4 Results ---\n');
P4_data_loaded = false; 
featureTableFileToLoad = '';
featureTableFile_dated = fullfile(resultsPath_P4, sprintf('%s_Phase4_FeatureImportanceTable_WithPvalues.csv', dateStrForFilenames));
altFeatureTableFiles = dir(fullfile(resultsPath_P4, '*_Phase4_FeatureImportanceTable_WithPvalues.csv'));
if exist(featureTableFile_dated, 'file')
    featureTableFileToLoad = featureTableFile_dated;
elseif ~isempty(altFeatureTableFiles)
    [~,idxSort_P4F] = sort([altFeatureTableFiles.datenum],'descend');
    featureTableFileToLoad = fullfile(resultsPath_P4, altFeatureTableFiles(idxSort_P4F(1)).name);
    fprintf('Note: Using latest available feature importance table: %s\n', featureTableFileToLoad);
else
     fprintf('Phase 4 feature importance table not found. Skipping Phase 4 visualizations.\n');
end

if ~isempty(featureTableFileToLoad)
    try
        P4_feature_importance_table = readtable(featureTableFileToLoad);
        P4_data_loaded = true;
        fprintf('Phase 4 Feature Importance Table loaded from: %s\n', featureTableFileToLoad);
    catch ME_P4F
        fprintf('ERROR loading Phase 4 feature importance table from %s: %s\n', featureTableFileToLoad, ME_P4F.message);
    end
end

%% 6. Generate Phase 4 Visualizations
% ... (This section remains unchanged) ...
fprintf('\n--- 6. Generating Phase 4 Visualizations ---\n');
if P4_data_loaded && P3_data_loaded && ~isempty(finalModelPackage)
    figP4Coeff = figure('Name', 'Phase 4 - LDA Coefficient Spectrum', 'Position', [100, 100, 900, 600]);
    if ismember('BinnedWavenumber_cm_neg1', P4_feature_importance_table.Properties.VariableNames) && ...
       ismember('LDACoefficient', P4_feature_importance_table.Properties.VariableNames) && ...
       isfield(finalModelPackage, 'binningFactor') 
        
        plot_wavenumbers_p4 = P4_feature_importance_table.BinnedWavenumber_cm_neg1;
        plot_coeffs_p4 = P4_feature_importance_table.LDACoefficient;
        [plot_wavenumbers_p4_sorted, sortIdx_p4] = sort(plot_wavenumbers_p4);
        plot_coeffs_p4_sorted = plot_coeffs_p4(sortIdx_p4);

        stem(plot_wavenumbers_p4_sorted, plot_coeffs_p4_sorted, 'filled', 'MarkerSize', 4);
        hold on; plot(plot_wavenumbers_p4_sorted, zeros(size(plot_wavenumbers_p4_sorted)), 'k--'); hold off;
        xlabel(sprintf('Binned Wavenumber (cm^{-1}) - Binning Factor %d', finalModelPackage.binningFactor)); 
        ylabel('LDA Coefficient Value'); 
        title({'LDA Coefficients for MRMR-Selected Features (WHO-1 vs WHO-3)'; 'Positive values indicate contribution towards WHO-3'},'FontWeight','Normal'); 
        grid on; ax_p4c = gca; ax_p4c.XDir = 'reverse'; 
        if ~isempty(plot_wavenumbers_p4_sorted), xlim([min(plot_wavenumbers_p4_sorted)-5 max(plot_wavenumbers_p4_sorted)+5]); else xlim([900 1800]); end 

        ldaCoeffPlotP4Filename = fullfile(figuresPath_output, sprintf('%s_P4_LDACoeffSpectrum.tiff', dateStrForFilenames));
        exportgraphics(figP4Coeff, ldaCoeffPlotP4Filename, 'Resolution', 300);
        savefig(figP4Coeff, strrep(ldaCoeffPlotP4Filename, '.tiff','.fig'));
        fprintf('Phase 4 LDA Coefficient Spectrum saved to: %s\n', ldaCoeffPlotP4Filename);
    else
        fprintf('Skipping LDA Coefficient plot: Required columns not in feature table or binningFactor missing from finalModelPackage.\n');
    end
    if isgraphics(figP4Coeff), close(figP4Coeff); end
    
    fprintf('Suggestion for Phase 4 Mean Spectra with Highlighted Features: This plot requires loading original training data (X_train_no_outliers, y_train_numeric_no_outliers) and original wavenumbers (wavenumbers_roi), applying the binning factor from finalModelPackage, then calculating mean spectra and highlighting features based on finalModelPackage.selectedWavenumbers and LDA coefficients. This logic is present in run_phase4_feature_interpretation.m (Section 4) and can be adapted or called as a function here if the data is made available to this script.\n');

else
    if ~P4_data_loaded, fprintf('Skipping Phase 4 visualizations: P4_feature_importance_table not loaded.\n'); end
    if ~P3_data_loaded || isempty(finalModelPackage)
        fprintf('Skipping Phase 4 visualizations: finalModelPackage from Phase 3 not loaded or empty (needed for binning factor context).\n'); 
    end
end



% ===== HELPER FUNCTION DEFINITIONS (MUST BE AT THE END OF THE SCRIPT FILE) =====

function makeBarChart(dataOR, dataAND, stdOR, stdAND, pipelineNames, metricName, titleStr, yLabelStr, figNameSuffix, outputPath, datePrefix, strategyPlotColors)
    fig = figure('Name', titleStr, 'Position', [100, 100, 1000, 600]);
    if isempty(dataAND) 
        bar_data = dataOR;
        b = bar(bar_data);
        hold on;
        errorbar(1:length(dataOR), dataOR, stdOR, 'k.', 'HandleVisibility','off');
        b(1).FaceColor = strategyPlotColors.OR;
        legendTxt = {'T2 OR Q Strategy'};
    else 
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
        b(1).FaceColor = strategyPlotColors.OR; 
        b(2).FaceColor = strategyPlotColors.AND; 
        legendTxt = {'T2 OR Q Strategy', 'T2 AND Q Strategy (Consensus)'};
    end
    hold off;
    xticks(1:length(pipelineNames)); xticklabels(pipelineNames); xtickangle(45);
    ylabel(yLabelStr); title({titleStr; '(Error bars: +/-1 Std.Dev. outer CV scores)'}, 'FontWeight', 'normal');
    legend(legendTxt, 'Location', 'NorthEastOutside'); grid on;
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

% Placed at the END of run_visualize_project_results.m
function makeTiledSpiderPlot(mean_metric_OR, std_metric_OR, mean_metric_AND, std_metric_AND, ...
                             pipelineNames, metricStr, axesLimits, axesInterval, figNameSuffix, ...
                             outputPath, datePrefix, strategyColors, shadeColors, plotFontSize_arg)

    if ~(exist('spider_plot_R2019b', 'file') || exist('spider_plot', 'file'))
        fprintf('Spider plot function not found. Skipping %s spider plot.\n', metricStr);
        return;
    end

    figSpider = figure('Name', sprintf('Spider Plots - %s per Pipeline by Strategy', metricStr), ...
                       'Position', [100, 100, 1100, 550]); 
    tl_spider = tiledlayout(1, 2, 'TileSpacing', 'loose', 'Padding', 'compact'); 
    sgtitle(tl_spider, sprintf('%s Performance Comparison by Outlier Strategy', metricStr), 'FontSize', plotFontSize_arg+2, 'FontWeight','Normal'); 

    spiderAxesLabels = pipelineNames'; 
    axesLabelsOffset_val = 0.3; 
    axesZoom_val = 0.7; % Reset zoom to default, as alignment might fix label issues. Adjust if still needed.

    % Settings for the numerical tick labels (0.6-1.0)
    axesTickFontColor = [0, 0, 0]; % Black for better visibility
    axesTickHorzAlign = 'right';   % Shift numerical labels to the right of the axis line
    axesTickVertAlign = 'middle';  % Keep them vertically centered on their web lines

    % Plot for OR Strategy
    ax_OR = nexttile(tl_spider);
    P_OR = mean_metric_OR'; 
    P_OR(isnan(P_OR)) = axesLimits(1); 
    
    lower_bound_OR = mean_metric_OR' - std_metric_OR';
    upper_bound_OR = mean_metric_OR' + std_metric_OR';
    lower_bound_OR = max(lower_bound_OR, axesLimits(1)); 
    upper_bound_OR = min(upper_bound_OR, axesLimits(2));
    shaded_limits_OR = {[lower_bound_OR; upper_bound_OR]}; 

    try
        if exist('spider_plot_R2019b', 'file')
            spider_plot_R2019b(P_OR, ...
                'AxesLabels', spiderAxesLabels, 'AxesLimits', axesLimits, ...
                'AxesInterval', axesInterval, 'AxesPrecision', 2, 'AxesDisplay', 'one', ...
                'AxesFontColor', axesTickFontColor, ...         % Updated tick label color
                'AxesHorzAlign', axesTickHorzAlign, ...     % Added for tick label horizontal alignment
                'AxesVertAlign', axesTickVertAlign, ...     % Added for tick label vertical alignment
                'AxesFontSize', plotFontSize_arg-1, ...
                'LabelFontSize', plotFontSize_arg, 'AxesLabelsOffset', axesLabelsOffset_val, ...
                'AxesZoom', axesZoom_val, ... 
                'FillOption', 'off', ... 
                'Color', strategyColors.OR, 'LineWidth', 2, ...
                'Marker', 'o', 'MarkerSize', 70, ...
                'AxesShaded', 'on', 'AxesShadedLimits', shaded_limits_OR, ... 
                'AxesShadedColor', {shadeColors.OR}, 'AxesShadedTransparency', 0.2);
        else 
            spider_plot(P_OR, 'AxesLabels', spiderAxesLabels, 'AxesLimits', axesLimits, ...
                'AxesInterval', axesInterval, 'AxesPrecision', 2, 'AxesDisplay', 'one',...
                'AxesFontColor', axesTickFontColor, ... % Updated
                'AxesHorzAlign', axesTickHorzAlign, ... % Added
                'AxesVertAlign', axesTickVertAlign, ... % Added
                'AxesLabelsOffset', axesLabelsOffset_val, ... 
                'AxesZoom', axesZoom_val, ... 
                'FillOption', 'off', 'Color', strategyColors.OR, 'LineWidth', 2); 
             warning('Using generic spider_plot.m for OR; std dev shading, zoom, and precise tick alignments might behave differently or not be supported as named options.');
        end
        title(ax_OR, 'T2 OR Q Strategy', 'FontWeight', 'Normal', 'FontSize', plotFontSize_arg); 
    catch ME_spider_OR
         fprintf('ERROR creating OR strategy spider plot for %s: %s\n', metricStr, ME_spider_OR.message);
    end

    % Plot for AND Strategy
    ax_AND = nexttile(tl_spider);
    P_AND = mean_metric_AND';
    P_AND(isnan(P_AND)) = axesLimits(1);
    
    lower_bound_AND = mean_metric_AND' - std_metric_AND';
    upper_bound_AND = mean_metric_AND' + std_metric_AND';
    lower_bound_AND = max(lower_bound_AND, axesLimits(1));
    upper_bound_AND = min(upper_bound_AND, axesLimits(2));
    shaded_limits_AND = {[lower_bound_AND; upper_bound_AND]};

    try
        if exist('spider_plot_R2019b', 'file')
            spider_plot_R2019b(P_AND, ...
                'AxesLabels', spiderAxesLabels, 'AxesLimits', axesLimits, ...
                'AxesInterval', axesInterval, 'AxesPrecision', 2, 'AxesDisplay', 'one', ...
                'AxesFontColor', axesTickFontColor, ...     % Updated tick label color
                'AxesHorzAlign', axesTickHorzAlign, ... % Added for tick label horizontal alignment
                'AxesVertAlign', axesTickVertAlign, ... % Added for tick label vertical alignment
                'AxesFontSize', plotFontSize_arg-1, ...
                'LabelFontSize', plotFontSize_arg, 'AxesLabelsOffset', axesLabelsOffset_val, ... 
                'AxesZoom', axesZoom_val, ... 
                'FillOption', 'off', ... 
                'Color', strategyColors.AND, 'LineWidth', 2, ...
                'Marker', 's', 'MarkerSize', 70, ...
                'AxesShaded', 'on', 'AxesShadedLimits', shaded_limits_AND, ... 
                'AxesShadedColor', {shadeColors.AND}, 'AxesShadedTransparency', 0.2);
        else 
            spider_plot(P_AND, 'AxesLabels', spiderAxesLabels, 'AxesLimits', axesLimits, ...
                'AxesInterval', axesInterval, 'AxesPrecision', 2, 'AxesDisplay', 'one', ...
                'AxesFontColor', axesTickFontColor, ... % Updated
                'AxesHorzAlign', axesTickHorzAlign, ... % Added
                'AxesVertAlign', axesTickVertAlign, ... % Added
                'AxesLabelsOffset', axesLabelsOffset_val, ... 
                'AxesZoom', axesZoom_val, ... 
                'FillOption', 'off', 'Color', strategyColors.AND, 'LineWidth', 2); 
            warning('Using generic spider_plot.m for AND; std dev shading, zoom, and precise tick alignments might behave differently or not be supported as named options.');
        end
        title(ax_AND, 'T2 AND Q Strategy (Consensus)', 'FontWeight', 'Normal','FontSize', plotFontSize_arg); 
    catch ME_spider_AND
        fprintf('ERROR creating AND strategy spider plot for %s: %s\n', metricStr, ME_spider_AND.message);
    end
    
    filenameBase_Spider = fullfile(outputPath, sprintf('%s_P2_SpiderPlot_%s_TiledStrategies', datePrefix, figNameSuffix));
    exportgraphics(figSpider, [filenameBase_Spider, '.tiff'], 'Resolution', 300);
    savefig(figSpider, [filenameBase_Spider, '.fig']);
    fprintf('Tiled spider plot for %s saved to: %s.(tiff/fig)\n', metricStr, filenameBase_Spider);
    if isgraphics(figSpider), close(figSpider); end
end