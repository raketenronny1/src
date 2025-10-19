function run_phase4_feature_interpretation(cfg)
%RUN_PHASE4_FEATURE_INTERPRETATION
%
% Script for Phase 4: Final Interpretation & Reporting.
% Focuses on identifying key discriminative wavenumbers from the best model
% and their importance as indicated by LDA coefficients.
%
% Date Modified: 2025-06-07 (Dynamic plot labeling based on best pipeline)
% Accepts either a configuration struct or a YAML file path.

%% 0. Initialization
% =========================================================================
fprintf('PHASE 4: Feature Interpretation - %s\n', string(datetime('now')));

if nargin < 1 || isempty(cfg)
    cfg = configure_cfg();
elseif ischar(cfg) || (isstring(cfg) && isscalar(cfg))
    cfg = configure_cfg('configFile', char(cfg));
elseif ~isstruct(cfg)
    error('run_phase4_feature_interpretation:InvalidConfig', ...
        'Configuration input must be empty, a struct or a file path.');
end
if ~isfield(cfg, 'projectRoot')
    cfg.projectRoot = pwd;
end

P = setup_project_paths(cfg.projectRoot, '', cfg); % Use helper
dataPath = P.dataPath;
resultsPath = fullfile(P.resultsPath, 'Phase4');
figuresPath = fullfile(P.figuresPath, 'Phase4');

if ~exist(resultsPath, 'dir'), mkdir(resultsPath); end
if ~exist(figuresPath, 'dir'), mkdir(figuresPath); end

dateStr = string(datetime('now','Format','yyyyMMdd'));

%% 1. Load Final Model Package from Phase 3
% =========================================================================
fprintf('Loading final model package from Phase 3...\n');
compFiles = dir(fullfile(P.resultsPath, 'Phase3', '*_Phase3_ComparisonResults.mat'));
if isempty(compFiles)
    error('No Phase 3 comparison results found. Run Phase 3 first.');
end
[~,idxSort] = sort([compFiles.datenum],'descend');
latestComp = load(fullfile(compFiles(idxSort(1)).folder, compFiles(idxSort(1)).name),'bestModelInfo');

% --- Get Best Pipeline Name and Model File ---
bestPipelineName = latestComp.bestModelInfo.name;
latestModelFile = latestComp.bestModelInfo.modelFile;
fprintf('== Best pipeline identified: %s ==\n', bestPipelineName);
fprintf('Using final model from: %s\n', latestModelFile);

load(latestModelFile,'finalModel');
finalModelPackage = finalModel;

% Extract necessary components
finalLDAModel = finalModelPackage.LDAModel;
selectedWavenumbers = finalModelPackage.selectedWavenumbers;
selectedFeatureIndices_in_binned = finalModelPackage.selectedFeatureIndices;

if isempty(selectedWavenumbers) || isempty(finalLDAModel.Coeffs)
    error('Selected wavenumbers or LDA coefficients not found in the loaded model package.');
end

fprintf('%d features (binned wavenumbers) were used by the final model.\n', length(selectedWavenumbers));


%% 2. Extract and Analyze LDA Coefficients
% =========================================================================
fprintf('\n--- Analyzing LDA Coefficients ---\n');

classNames = finalLDAModel.ClassNames; 
fprintf('LDA Model Class Names: %s\n', mat2str(classNames));
idx_who3_in_model = find(classNames == 3);
idx_who1_in_model = find(classNames == 1);
if isempty(idx_who3_in_model) || isempty(idx_who1_in_model)
    error('Could not find both WHO-1 and WHO-3 in the LDA model class names.');
end

ldaCoefficients = [];
if idx_who3_in_model == 2 
    ldaCoefficients = finalLDAModel.Coeffs(1,2).Linear;
    fprintf('Interpreting positive LDA coefficients as indicative of WHO-3.\n');
elseif idx_who1_in_model == 2 
    ldaCoefficients = -finalLDAModel.Coeffs(1,2).Linear;
    fprintf('Interpreting NEGATIVE of Coeffs(1,2).Linear as indicative of WHO-3.\n');
else
    error('Unexpected class order or setup in LDA model coefficients.');
end

if length(ldaCoefficients) ~= length(selectedWavenumbers)
    error('Mismatch between number of LDA coefficients (%d) and number of selected features (%d).', length(ldaCoefficients), length(selectedWavenumbers));
end

directionInfluence = strings(length(ldaCoefficients), 1);
directionInfluence(ldaCoefficients > 0) = "Higher in WHO-3";
directionInfluence(ldaCoefficients < 0) = "Lower in WHO-3 (Higher in WHO-1)";
directionInfluence(ldaCoefficients == 0) = "No linear influence";

featureImportanceTable = table(...
    selectedWavenumbers(:), ... 
    ldaCoefficients(:), ...   
    directionInfluence, ...
    'VariableNames', {'BinnedWavenumber_cm_neg1', 'LDACoefficient', 'DirectionOfInfluence_WHO3'});


%% 2.5 Prepare Training Data for Visualization
% =========================================================================
fprintf('\n--- Preparing training data for Phase 4 visualization ---\n');
trainTbl = load(fullfile(dataPath,'data_table_train.mat'),'dataTableTrain');
wData = load(fullfile(dataPath,'wavenumbers.mat'),'wavenumbers_roi');
X_train_full_for_plot = vertcat(trainTbl.dataTableTrain.CombinedSpectra{:});
y_train_full_for_plot = arrayfun(@(g) double(strcmp(g,'WHO-3'))*2+1, trainTbl.dataTableTrain.WHO_Grade, 'UniformOutput', false);
y_train_full_for_plot = vertcat(y_train_full_for_plot{:});
wavenumbers_original = wData.wavenumbers_roi;
if iscolumn(wavenumbers_original); wavenumbers_original = wavenumbers_original'; end

binningFactorForPlot = finalModelPackage.binningFactor;
if binningFactorForPlot > 1
    [X_train_binned_for_plot, wavenumbers_binned_for_plot] = bin_spectra(X_train_full_for_plot, wavenumbers_original, binningFactorForPlot);
else
    X_train_binned_for_plot = X_train_full_for_plot;
    wavenumbers_binned_for_plot = wavenumbers_original;
end

[~, sortIdxCoeff] = sort(abs(featureImportanceTable.LDACoefficient), 'descend');
sortedFeatureImportanceTable = featureImportanceTable(sortIdxCoeff,:);
fprintf('\nTop 10 Most Influential Features (Binned Wavenumbers):\n');
disp(sortedFeatureImportanceTable(1:min(10, height(sortedFeatureImportanceTable)),:));
featureTableFilename = fullfile(resultsPath, sprintf('%s_Phase4_FeatureImportanceTable.csv', dateStr));
writetable(sortedFeatureImportanceTable, featureTableFilename);
fprintf('Full feature importance table saved to: %s\n', featureTableFilename);


%% 3. Plot LDA Coefficients (Feature Importance Spectrum)
% =========================================================================
fprintf('\n--- Plotting LDA Coefficient Spectrum ---\n');
figure('Name', 'LDA Coefficient Spectrum for Selected Features', 'Position', [100, 100, 900, 600]);
[plot_wavenumbers_from_table, sortIdxWn_plot] = sort(featureImportanceTable.BinnedWavenumber_cm_neg1);
plot_coeffs_from_table = featureImportanceTable.LDACoefficient(sortIdxWn_plot);

stem(plot_wavenumbers_from_table, plot_coeffs_from_table, 'filled', 'MarkerSize', 4);
hold on;
plot(plot_wavenumbers_from_table, zeros(size(plot_wavenumbers_from_table)), 'k--');
hold off;

xlabel(sprintf('Binned Wavenumber (cm^{-1}) - Binning Factor %d', finalModelPackage.binningFactor));
ylabel('LDA Coefficient Value');

% --- DYNAMIC TITLE ---
dynamicTitle_Coeffs = sprintf('LDA Coefficients for %s-Selected Features (WHO-1 vs WHO-3)', bestPipelineName);
title({dynamicTitle_Coeffs, 'Positive values indicate contribution towards WHO-3'});
% --- END DYNAMIC TITLE ---

grid on;
ax = gca;
ax.XDir = 'reverse'; 
if ~isempty(plot_wavenumbers_from_table)
    xlim([min(plot_wavenumbers_from_table)-5 max(plot_wavenumbers_from_table)+5 ]);
else
    xlim([900 1800]); 
end

ldaCoeffPlotFilenameBase = fullfile(figuresPath, sprintf('%s_Phase4_LDACoeffSpectrum_%s', dateStr, bestPipelineName));
savefig(gcf, [ldaCoeffPlotFilenameBase, '.fig']);
exportgraphics(gcf, [ldaCoeffPlotFilenameBase, '.tiff'], 'Resolution', 300);
fprintf('LDA coefficient spectrum plot saved to: %s.(fig/tiff)\n', ldaCoeffPlotFilenameBase);

%% 4. Plot Mean Spectra with Highlighted Important Regions
% =========================================================================
fprintf('\n--- Generating Mean Spectra Plot with Highlighted Key Features ---\n');

if exist('X_train_binned_for_plot', 'var') && ~isempty(X_train_binned_for_plot)
    y_train_for_mean_spectra = y_train_full_for_plot;

    mean_spectrum_who1_binned = mean(X_train_binned_for_plot(y_train_for_mean_spectra==1, :), 1);
    mean_spectrum_who3_binned = mean(X_train_binned_for_plot(y_train_for_mean_spectra==3, :), 1);
    difference_spectrum_binned = mean_spectrum_who3_binned - mean_spectrum_who1_binned;
    
    % --- DYNAMIC FIGURE NAME ---
    figure('Name', sprintf('Mean Spectra & Difference with Highlighted %s Features', bestPipelineName), 'Position', [100, 100, 900, 700]);
    colorWHO1 = [0.9, 0.6, 0.4]; colorWHO3 = [0.4, 0.702, 0.902];
    colorDiff = [0.3 0.3 0.3]; colorMarkerIncreaseWHO3 = [0.8 0 0];
    colorMarkerDecreaseWHO3 = [0 0.6 0];

    % --- Subplot 1: Mean Spectra ---
    ax1 = subplot(2,1,1);
    hold(ax1, 'on');
    legend_handles = [];
    h_who1 = plot(ax1, wavenumbers_binned_for_plot, mean_spectrum_who1_binned, 'Color', colorWHO1, 'LineWidth', 1.5, 'DisplayName', 'Mittelwert WHO-1');
    legend_handles = [legend_handles, h_who1];
    h_who3 = plot(ax1, wavenumbers_binned_for_plot, mean_spectrum_who3_binned, 'Color', colorWHO3, 'LineWidth', 1.5, 'DisplayName', 'Mittelwert WHO-3');
    legend_handles = [legend_handles, h_who3];
    
    selectedWnsFromModel = finalModelPackage.selectedWavenumbers;
    ldaCoeffsFromModel = ldaCoefficients; 

    for k_idx = 1:length(selectedWnsFromModel)
        current_selected_wn = selectedWnsFromModel(k_idx);
        plot_idx = find(abs(wavenumbers_binned_for_plot - current_selected_wn) < 1e-3, 1);
        if ~isempty(plot_idx)
            marker_y_pos_who3 = mean_spectrum_who3_binned(plot_idx);
            marker_y_pos_who1 = mean_spectrum_who1_binned(plot_idx);
            marker_color = [0.5 0.5 0.5];
            if ldaCoeffsFromModel(k_idx) > 0, marker_color = colorMarkerIncreaseWHO3;
            elseif ldaCoeffsFromModel(k_idx) < 0, marker_color = colorMarkerDecreaseWHO3; end
            plot(ax1, wavenumbers_binned_for_plot(plot_idx), marker_y_pos_who3, 'v', 'MarkerFaceColor', marker_color, 'MarkerEdgeColor', 'k', 'MarkerSize', 7, 'HandleVisibility', 'off');
            plot(ax1, wavenumbers_binned_for_plot(plot_idx), marker_y_pos_who1, '^', 'MarkerFaceColor', marker_color, 'MarkerEdgeColor', 'k', 'MarkerSize', 7, 'HandleVisibility', 'off');
        end
    end
    
    % --- DYNAMIC TITLE ---
    title(ax1, sprintf('Mittlere Spektren (Gebinnt, Faktor %d) mit %s Merkmalen', binningFactorForPlot, bestPipelineName));
    xlabel(ax1, 'Wellenzahl (cm^{-1})'); ylabel(ax1, 'Mittlere Absorption (A.U.)');
    ax1.XDir = 'reverse'; grid on;
    if ~isempty(wavenumbers_binned_for_plot), xlim(ax1, [min(wavenumbers_binned_for_plot) max(wavenumbers_binned_for_plot)]); end

    h_inc = plot(ax1, NaN,NaN,'v', 'MarkerFaceColor', colorMarkerIncreaseWHO3, 'MarkerEdgeColor','k', 'MarkerSize',7, 'DisplayName', 'Merkmal ↑ WHO-3');
    h_dec = plot(ax1, NaN,NaN,'^', 'MarkerFaceColor', colorMarkerDecreaseWHO3, 'MarkerEdgeColor','k', 'MarkerSize',7, 'DisplayName', 'Merkmal ↓ WHO-3');
    legend_handles = [legend_handles, h_inc, h_dec];
    hold(ax1, 'off');
    legend(ax1, legend_handles, 'Location','NorthWest');
    
    % --- Subplot 2: Difference Spectrum ---
    ax2 = subplot(2,1,2);
    plot(ax2, wavenumbers_binned_for_plot, difference_spectrum_binned, 'Color', colorDiff, 'LineWidth', 1.5, 'DisplayName', 'Differenz (WHO-3 minus WHO-1)');
    hold(ax2, 'on');
    
    for k_idx = 1:length(selectedWnsFromModel)
        current_selected_wn = selectedWnsFromModel(k_idx);
        plot_idx = find(abs(wavenumbers_binned_for_plot - current_selected_wn) < 1e-3, 1);
        if ~isempty(plot_idx)
            marker_y_pos_diff = difference_spectrum_binned(plot_idx);
            marker_color = [0.5 0.5 0.5];
            if ldaCoeffsFromModel(k_idx) > 0, marker_color = colorMarkerIncreaseWHO3;
            elseif ldaCoeffsFromModel(k_idx) < 0, marker_color = colorMarkerDecreaseWHO3; end
            plot(ax2, wavenumbers_binned_for_plot(plot_idx), marker_y_pos_diff, 'o', 'MarkerFaceColor', marker_color, 'MarkerEdgeColor', 'k', 'MarkerSize', 7, 'HandleVisibility', 'off');
        end
    end
    plot(ax2, wavenumbers_binned_for_plot, zeros(size(wavenumbers_binned_for_plot)), 'k:', 'HandleVisibility','off');
    hold(ax2, 'off');
    
    % --- DYNAMIC TITLE ---
    title(ax2, sprintf('Differenzspektrum mit %s Merkmalen', bestPipelineName));
    xlabel(ax2, 'Wellenzahl (cm^{-1})'); ylabel(ax2, 'Differenz der Absorption');
    ax2.XDir = 'reverse'; legend(ax2, 'show', 'Location','SouthWest'); grid on;
    if ~isempty(wavenumbers_binned_for_plot), xlim(ax2, [min(wavenumbers_binned_for_plot) max(wavenumbers_binned_for_plot)]); end   
    
    sgtitle('Vergleich der mittleren Spektren und hervorgehobene Merkmale', 'FontSize', 14, 'FontWeight', 'bold');

    meanSpectraPlotFilenameBase = fullfile(figuresPath, sprintf('%s_Phase4_MeanSpectra_Highlighted_%s', dateStr, bestPipelineName));
    savefig(gcf, [meanSpectraPlotFilenameBase, '.fig']);
    exportgraphics(gcf, [meanSpectraPlotFilenameBase, '.tiff'], 'Resolution', 300);
    fprintf('Mean spectra plot with highlighted features saved to: %s.(fig/tiff)\n', meanSpectraPlotFilenameBase);
else
    fprintf('Skipping mean spectra plot due to missing training data for plotting.\n');
end

end