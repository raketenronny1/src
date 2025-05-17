% run_exploratory_outlier_analysis.m
%
% PURPOSE:
%   To perform an exploratory analysis of potential outliers in the 
%   training dataset using PCA, Hotelling's T-squared, and Q-residuals.
%   This script focuses on generating visualizations and statistics to
%   inform subsequent outlier removal decisions. 
%   NO DATA IS REMOVED by this script.
%
% INPUTS (expected in workspace or loaded from .mat):
%   - dataTableTrain: Table containing probe-level training data, with at least
%                     'CombinedSpectra', 'WHO_Grade', and 'Diss_ID' columns.
%                     'CombinedSpectra' should contain the preprocessed spectra
%                     (e.g., after SG smoothing, SNV, L2-norm).
%   - wavenumbers_roi: Vector of wavenumber values.
%
% OUTPUTS (saved to files):
%   - Various diagnostic plots (.tiff and .fig).
%   - A .mat file containing T2_values, Q_values, thresholds, flags for
%     potential outliers, PCA model info, and spectrum identifiers.
%
% DATE: 2025-05-17

%% --- 0. Configuration & Setup ---
% clear; clc; close all; % Optional
fprintf('Starting Exploratory Outlier Analysis - %s\n', string(datetime('now')));

% --- Define Paths (User to verify/modify) ---
projectBasePath = 'C:\Users\Franz\OneDrive\01_Promotion\01 Data\meningioma-ftir-classification'; 
dataDir    = fullfile(projectBasePath, 'data');
resultsDir = fullfile(projectBasePath, 'results', 'Phase1_ExploratoryOutlierAnalysis');
figuresDir = fullfile(projectBasePath, 'figures', 'Phase1_ExploratoryOutlierAnalysis');

if ~isfolder(resultsDir); mkdir(resultsDir); end
if ~isfolder(figuresDir); mkdir(figuresDir); end

% --- Parameters ---
P.alpha_significance = 0.05;
P.variance_to_explain_for_PCA_model = 0.95; % For k_model determination
P.datePrefix = string(datetime('now','Format','yyyyMMdd'));

% --- Plotting Defaults ---
P.colorWHO1 = [0.9, 0.6, 0.4]; 
P.colorWHO3 = [0.4, 0.702, 0.902]; 
P.colorOutlierT2 = [0.8, 0.2, 0.2];   
P.colorOutlierQ = [0.2, 0.2, 0.8];    
P.colorOutlierBoth = [0.8, 0, 0.8]; 
P.colorOutlierHighlight = [1 0.5 0]; 

P.plotFontSize = 10;
P.plotXLabel = 'Wellenzahl (cm^{-1})';
P.plotYLabelAbsorption = 'Absorption (a.u.)';
P.plotXLim = [950 1800];

fprintf('Setup complete. Results will be saved in: \n  %s\n  %s\n', resultsDir, figuresDir);

%% --- 1. Load Training Data ---
fprintf('\n--- 1. Loading Training Data ---\n');
trainDataTableFile = fullfile(dataDir, 'data_table_train.mat'); 
if exist(trainDataTableFile, 'file')
    fprintf('Loading dataTableTrain from: %s\n', trainDataTableFile);
    loadedVars = load(trainDataTableFile);
    if isfield(loadedVars, 'dataTableTrain')
        dataTableTrain = loadedVars.dataTableTrain;
        fprintf('dataTableTrain loaded with %d probes.\n', height(dataTableTrain));
    else
        error('Variable "dataTableTrain" not found within %s.', trainDataTableFile);
    end
else
    error('Input file %s not found. Please run data splitting script first.', trainDataTableFile);
end
try
    load(fullfile(dataDir, 'wavenumbers.mat'), 'wavenumbers_roi');
    if iscolumn(wavenumbers_roi), wavenumbers_roi = wavenumbers_roi'; end
catch ME_wave
    error('Error loading wavenumbers.mat: %s', ME_wave.message);
end
fprintf('wavenumbers_roi loaded (%d points).\n', length(wavenumbers_roi));

allSpectra_cell = {}; allLabels_cell = {}; allPatientIDs_cell = {};
allOriginalProbeRowIndices_vec = []; allOriginalSpectrumIndices_InProbe_vec = []; 
fprintf('Extracting and flattening spectra from dataTableTrain.CombinedSpectra...\n');
for i = 1:height(dataTableTrain)
    spectraMatrix = dataTableTrain.CombinedSpectra{i}; 
    if isempty(spectraMatrix) || ~isnumeric(spectraMatrix) || ndims(spectraMatrix) ~= 2, warning('Probe %s (Row %d): CombinedSpectra is invalid/empty. Skipping.', dataTableTrain.Diss_ID{i}, i); continue; end
    if size(spectraMatrix,1) == 0, warning('Probe %s (Row %d): CombinedSpectra has zero spectra. Skipping.', dataTableTrain.Diss_ID{i}, i); continue; end
    if size(spectraMatrix,2) ~= length(wavenumbers_roi), warning('Probe %s (Row %d): Wavenumber mismatch. Skipping.', dataTableTrain.Diss_ID{i}, i); continue; end
    numIndividualSpectra = size(spectraMatrix, 1);
    allSpectra_cell{end+1,1} = spectraMatrix;
    currentLabel = dataTableTrain.WHO_Grade(i);
    allLabels_cell{end+1,1} = repmat(currentLabel, numIndividualSpectra, 1);
    currentDissID = dataTableTrain.Diss_ID{i}; 
    allPatientIDs_cell{end+1,1} = repmat(currentDissID, numIndividualSpectra, 1);
    allOriginalProbeRowIndices_vec = [allOriginalProbeRowIndices_vec; repmat(i, numIndividualSpectra, 1)];
    allOriginalSpectrumIndices_InProbe_vec = [allOriginalSpectrumIndices_InProbe_vec; (1:numIndividualSpectra)'];
end
if isempty(allSpectra_cell), error('No valid spectra extracted.'); end
X_train_explore = cell2mat(allSpectra_cell);
y_train_cat_explore = cat(1, allLabels_cell{:});
Patient_ID_explore = vertcat(allPatientIDs_cell{:}); 
Original_ProbeRowIndices_explore = allOriginalProbeRowIndices_vec; 
Original_SpectrumIndexInProbe_explore = allOriginalSpectrumIndices_InProbe_vec;
y_train_numeric_explore = zeros(length(y_train_cat_explore), 1);
categories_y = categories(y_train_cat_explore);
idx_who1 = find(strcmp(categories_y, 'WHO-1')); idx_who3 = find(strcmp(categories_y, 'WHO-3'));
if ~isempty(idx_who1), y_train_numeric_explore(y_train_cat_explore == categories_y{idx_who1}) = 1; end
if ~isempty(idx_who3), y_train_numeric_explore(y_train_cat_explore == categories_y{idx_who3}) = 3; end
fprintf('Data for exploration: %d spectra, %d features.\n', size(X_train_explore,1), size(X_train_explore,2));

%% --- 2. Perform PCA, Calculate T² and Q Statistics & Thresholds ---
fprintf('\n--- 2. PCA, T2/Q Calculation & Thresholds ---\n');
[coeff_explore, score_explore, latent_explore, ~, explained_explore, mu_explore] = ...
    pca(X_train_explore, 'Algorithm','svd');
fprintf('PCA completed. Variance by 1st PC: %.2f%%\n', explained_explore(1));
cumulativeVariance = cumsum(explained_explore);
k_model_explore = find(cumulativeVariance >= P.variance_to_explain_for_PCA_model*100, 1, 'first');
if isempty(k_model_explore), k_model_explore = min(length(explained_explore), size(X_train_explore,1)-1); end % Ensure k_model <= N-1
if k_model_explore == 0 && ~isempty(explained_explore), k_model_explore = 1; end
if isempty(explained_explore) || k_model_explore == 0, error('Could not determine k_model.'); end
fprintf('k_model (for T2/Q) based on >=%.0f%% variance: %d PCs.\n', P.variance_to_explain_for_PCA_model*100, k_model_explore);

score_k_model = score_explore(:, 1:k_model_explore);
lambda_k_model = latent_explore(1:k_model_explore); 
lambda_k_model(lambda_k_model <= eps) = eps;
T2_values_explore = sum(bsxfun(@rdivide, score_k_model.^2, lambda_k_model'), 2);
n_samples_explore = size(X_train_explore, 1);
if n_samples_explore > k_model_explore && k_model_explore > 0
    T2_threshold_explore = ((k_model_explore * (n_samples_explore - 1)) / (n_samples_explore - k_model_explore)) ...
        * finv(1 - P.alpha_significance, k_model_explore, n_samples_explore - k_model_explore);
else
    T2_threshold_explore = chi2inv(1 - P.alpha_significance, k_model_explore);
end
fprintf('Hotelling T2 threshold: %.4f\n', T2_threshold_explore);

X_reconstructed_explore = score_explore(:, 1:k_model_explore) * coeff_explore(:, 1:k_model_explore)' + mu_explore;
E_residuals_explore = X_train_explore - X_reconstructed_explore;
Q_values_explore = sum(E_residuals_explore.^2, 2);
num_total_pcs_latent = length(latent_explore);
Q_threshold_explore = NaN;
if k_model_explore < num_total_pcs_latent
    discarded_eigenvalues = latent_explore(k_model_explore+1:end);
    discarded_eigenvalues(discarded_eigenvalues <= eps) = eps;
    theta1 = sum(discarded_eigenvalues); theta2 = sum(discarded_eigenvalues.^2); theta3 = sum(discarded_eigenvalues.^3);
    if theta1 > eps && theta2 > eps
        h0 = 1 - (2 * theta1 * theta3) / (3 * theta2^2);
        if h0 <= eps, h0 = 1; end
        ca = norminv(1 - P.alpha_significance);
        val_in_bracket = ca * sqrt(2*theta2*h0^2)/theta1 + 1 + theta2*h0*(h0-1)/(theta1^2);
        if val_in_bracket > 0, Q_threshold_explore = theta1 * (val_in_bracket)^(1/h0); end
    end
end
if isnan(Q_threshold_explore) || isinf(Q_threshold_explore)
    Q_threshold_explore = prctile(Q_values_explore, (1-P.alpha_significance)*100);
    fprintf('Using empirical Q-threshold: %.4g\n', Q_threshold_explore);
else
    fprintf('Q-Statistic (SPE) threshold: %.4g\n', Q_threshold_explore);
end

flag_T2_outlier_explore = (T2_values_explore > T2_threshold_explore);
flag_Q_outlier_explore  = (Q_values_explore > Q_threshold_explore);
flag_AND_outlier_explore = flag_T2_outlier_explore & flag_Q_outlier_explore;
flag_OR_outlier_explore = flag_T2_outlier_explore | flag_Q_outlier_explore;
fprintf('Potential T2: %d. Q: %d. AND: %d. OR: %d\n', sum(flag_T2_outlier_explore), sum(flag_Q_outlier_explore), sum(flag_AND_outlier_explore), sum(flag_OR_outlier_explore));

%% --- 3. Generate Diagnostic Visualizations ---
fprintf('\n--- 3. Generating Diagnostic Visualizations ---\n');

% Plot 3.1: T² Values & Plot 3.2: Q Values (Individual plots as before)
% ... (Code for these plots from previous response - they are good for basic overview) ...
% (Ensure they use _explore suffixed variables)
% Example for T2 plot:
fig_t2 = figure('Name', 'Exploratory: Hotelling T2 Values');
plot(1:n_samples_explore, T2_values_explore, 'o', 'Color', [0.7 0.7 0.7], 'MarkerSize', 3, 'DisplayName', 'T2 Values'); hold on;
plot(find(flag_T2_outlier_explore), T2_values_explore(flag_T2_outlier_explore), 'x', 'Color', P.colorOutlierT2, 'MarkerSize', 5, 'DisplayName', 'T2 Outlier');
yline(T2_threshold_explore, '--r', sprintf('T2 Thresh (%.0f%%)', (1-P.alpha_significance)*100), 'LineWidth', 1.5);
hold off; xlabel('Spectrum Index'); ylabel('Hotelling T^2 Value'); title('Exploratory Hotelling T^2 Values'); legend show; grid on;
exportgraphics(fig_t2, fullfile(figuresDir, sprintf('%s_Exploratory_T2_Values.tiff', P.datePrefix)), 'Resolution', 300); savefig(fig_t2, fullfile(figuresDir, sprintf('%s_Exploratory_T2_Values.fig', P.datePrefix)));

fig_q = figure('Name', 'Exploratory: Q-Statistic Values');
plot(1:n_samples_explore, Q_values_explore, 'o', 'Color', [0.7 0.7 0.7], 'MarkerSize', 3, 'DisplayName', 'Q Values'); hold on;
plot(find(flag_Q_outlier_explore), Q_values_explore(flag_Q_outlier_explore), 'x', 'Color', P.colorOutlierQ, 'MarkerSize', 5, 'DisplayName', 'Q Outlier');
yline(Q_threshold_explore, '--m', sprintf('Q Thresh (%.0f%%)', (1-P.alpha_significance)*100), 'LineWidth', 1.5);
hold off; xlabel('Spectrum Index'); ylabel('Q-Statistic (SPE)'); title('Exploratory Q-Statistic Values'); legend show; grid on;
exportgraphics(fig_q, fullfile(figuresDir, sprintf('%s_Exploratory_Q_Values.tiff', P.datePrefix)), 'Resolution', 300); savefig(fig_q, fullfile(figuresDir, sprintf('%s_Exploratory_Q_Values.fig', P.datePrefix)));


% Plot 3.3: T² vs. Q Plot (Critical for identifying categories)
% (Code for this plot from previous response, ensure use of _explore vars)
fig_t2q = figure('Name', 'Exploratory: T2 vs Q Plot'); fig_t2q.Position = [100 100 700 550]; hold on;
is_normal = ~flag_T2_outlier_explore & ~flag_Q_outlier_explore;
is_T2_only = flag_T2_outlier_explore & ~flag_Q_outlier_explore;
is_Q_only = ~flag_T2_outlier_explore & flag_Q_outlier_explore;
is_T2_and_Q = flag_AND_outlier_explore;
h_plots_t2q = []; legend_texts_t2q = {};
if any(is_normal & y_train_numeric_explore==1), h_plots_t2q(end+1) = plot(T2_values_explore(is_normal & y_train_numeric_explore==1), Q_values_explore(is_normal & y_train_numeric_explore==1), 'o', 'Color', P.colorWHO1, 'MarkerSize', 4, 'MarkerFaceColor', P.colorWHO1); legend_texts_t2q{end+1} = 'WHO-1 (Normal)'; end
if any(is_normal & y_train_numeric_explore==3), h_plots_t2q(end+1) = plot(T2_values_explore(is_normal & y_train_numeric_explore==3), Q_values_explore(is_normal & y_train_numeric_explore==3), 'o', 'Color', P.colorWHO3, 'MarkerSize', 4, 'MarkerFaceColor', P.colorWHO3); legend_texts_t2q{end+1} = 'WHO-3 (Normal)'; end
if any(is_T2_only), h_plots_t2q(end+1) = plot(T2_values_explore(is_T2_only), Q_values_explore(is_T2_only), 's', 'Color', P.colorOutlierT2, 'MarkerSize', 5); legend_texts_t2q{end+1} = 'T2-only Outlier'; end
if any(is_Q_only), h_plots_t2q(end+1) = plot(T2_values_explore(is_Q_only), Q_values_explore(is_Q_only), 'd', 'Color', P.colorOutlierQ, 'MarkerSize', 5); legend_texts_t2q{end+1} = 'Q-only Outlier'; end
if any(is_T2_and_Q), h_plots_t2q(end+1) = plot(T2_values_explore(is_T2_and_Q), Q_values_explore(is_T2_and_Q), '*', 'Color', P.colorOutlierBoth, 'MarkerSize', 6); legend_texts_t2q{end+1} = 'T2 & Q Outlier'; end
line([T2_threshold_explore, T2_threshold_explore], get(gca,'YLim'), 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1, 'HandleVisibility','off');
line(get(gca,'XLim'), [Q_threshold_explore, Q_threshold_explore], 'Color', 'm', 'LineStyle', '--', 'LineWidth', 1, 'HandleVisibility','off');
hold off; xlabel(sprintf('Hotelling T^2 (Thresh: %.2f)', T2_threshold_explore)); ylabel(sprintf('Q-Statistic (SPE) (Thresh: %.2g)', Q_threshold_explore));
title(sprintf('T^2 vs. Q Plot (k_{model}=%d PCs)', k_model_explore), 'FontWeight','normal');
if ~isempty(h_plots_t2q), legend(h_plots_t2q, legend_texts_t2q, 'Location', 'best', 'FontSize', P.plotFontSize-1); end
grid on; xlim_max_t2q = max(T2_threshold_explore*1.5, prctile(T2_values_explore, 99.8)); ylim_max_t2q = max(Q_threshold_explore*1.5, prctile(Q_values_explore, 99.8));
if xlim_max_t2q > 0, xlim([0 xlim_max_t2q]); end; if ylim_max_t2q > 0, ylim([0 ylim_max_t2q]); end
exportgraphics(fig_t2q, fullfile(figuresDir, sprintf('%s_Exploratory_T2_vs_Q_Plot.tiff', P.datePrefix)), 'Resolution', 300); savefig(fig_t2q, fullfile(figuresDir, sprintf('%s_Exploratory_T2_vs_Q_Plot.fig', P.datePrefix)));


% Plot 3.4: PCA Score Plots (PC1vsPC2, PC1vsPC3, PC2vsPC3)
pcs_to_plot = [1,2; 1,3; 2,3]; % Combinations to plot
if size(score_explore,2) < 2
    pcs_to_plot = [1,1]; % Fallback for 1 PC data
elseif size(score_explore,2) < 3
    pcs_to_plot = [1,2]; % Only PC1 vs PC2 if only 2 PCs
end

for k_pc_plot = 1:size(pcs_to_plot,1)
    pc_x = pcs_to_plot(k_pc_plot, 1);
    pc_y = pcs_to_plot(k_pc_plot, 2);
    
    if pc_x > size(score_explore,2) || pc_y > size(score_explore,2), continue; end % Skip if not enough PCs

    fig_pca_pair = figure('Name', sprintf('Exploratory: PCA Scores PC%d vs PC%d', pc_x, pc_y));
    fig_pca_pair.Position = [150+k_pc_plot*20 150+k_pc_plot*20 700 550];
    hold on;
    h_pca_plots_pair = []; pca_legend_texts_pair = {};
    if any(is_normal & y_train_numeric_explore==1), h_pca_plots_pair(end+1) = scatter(score_explore(is_normal & y_train_numeric_explore==1, pc_x), score_explore(is_normal & y_train_numeric_explore==1, pc_y), 20, P.colorWHO1, 'o', 'filled', 'MarkerFaceAlpha', 0.4); pca_legend_texts_pair{end+1} = 'WHO-1 (Normal)'; end
    if any(is_normal & y_train_numeric_explore==3), h_pca_plots_pair(end+1) = scatter(score_explore(is_normal & y_train_numeric_explore==3, pc_x), score_explore(is_normal & y_train_numeric_explore==3, pc_y), 20, P.colorWHO3, 'o', 'filled', 'MarkerFaceAlpha', 0.4); pca_legend_texts_pair{end+1} = 'WHO-3 (Normal)'; end
    if any(is_T2_only), h_pca_plots_pair(end+1) = scatter(score_explore(is_T2_only,pc_x), score_explore(is_T2_only,pc_y), 30, P.colorOutlierT2, 's'); pca_legend_texts_pair{end+1} = 'T2-only Outlier'; end
    if any(is_Q_only), h_pca_plots_pair(end+1) = scatter(score_explore(is_Q_only,pc_x), score_explore(is_Q_only,pc_y), 30, P.colorOutlierQ, 'd'); pca_legend_texts_pair{end+1} = 'Q-only Outlier'; end
    if any(is_T2_and_Q), h_pca_plots_pair(end+1) = scatter(score_explore(is_T2_and_Q,pc_x), score_explore(is_T2_and_Q,pc_y), 35, P.colorOutlierBoth, '*'); pca_legend_texts_pair{end+1} = 'T2 & Q Outlier'; end
    hold off;
    xlabel(sprintf('PC%d (%.2f%%)', pc_x, explained_explore(pc_x)));
    if pc_x == pc_y && size(score_explore,2) == 1
         ylabel('(Only 1 PC available)');
    else
        ylabel(sprintf('PC%d (%.2f%%)', pc_y, explained_explore(pc_y)));
    end
    title(sprintf('PCA Score Plot (PC%d vs PC%d) - Exploratory',pc_x, pc_y), 'FontWeight','normal');
    if ~isempty(h_pca_plots_pair), legend(h_pca_plots_pair, pca_legend_texts_pair, 'Location', 'best', 'FontSize', P.plotFontSize-1); end
    axis equal; grid on; set(gca, 'FontSize', P.plotFontSize);
    exportgraphics(fig_pca_pair, fullfile(figuresDir, sprintf('%s_Exploratory_PCA_Scores_PC%dvsPC%d.tiff', P.datePrefix, pc_x, pc_y)), 'Resolution', 300);
    savefig(fig_pca_pair, fullfile(figuresDir, sprintf('%s_Exploratory_PCA_Scores_PC%dvsPC%d.fig', P.datePrefix, pc_x, pc_y)));
end

% Plot 3.5: 3D PCA Score Plot (PC1 vs PC2 vs PC3)
if size(score_explore,2) >= 3
    fig_pca_3d = figure('Name', 'Exploratory: PCA Score Plot 3D (PC1-3)');
    fig_pca_3d.Position = [200 200 750 600];
    ax3D = axes(fig_pca_3d); hold(ax3D, 'on');
    h_pca_plots_3d = []; pca_legend_texts_3d = {};
    if any(is_normal & y_train_numeric_explore==1), h_pca_plots_3d(end+1) = scatter3(ax3D, score_explore(is_normal & y_train_numeric_explore==1, 1), score_explore(is_normal & y_train_numeric_explore==1, 2), score_explore(is_normal & y_train_numeric_explore==1, 3), 20, P.colorWHO1, 'o', 'filled', 'MarkerFaceAlpha', 0.4); pca_legend_texts_3d{end+1} = 'WHO-1 (Normal)'; end
    if any(is_normal & y_train_numeric_explore==3), h_pca_plots_3d(end+1) = scatter3(ax3D, score_explore(is_normal & y_train_numeric_explore==3, 1), score_explore(is_normal & y_train_numeric_explore==3, 2), score_explore(is_normal & y_train_numeric_explore==3, 3), 20, P.colorWHO3, 'o', 'filled', 'MarkerFaceAlpha', 0.4); pca_legend_texts_3d{end+1} = 'WHO-3 (Normal)'; end
    if any(is_T2_only), h_pca_plots_3d(end+1) = scatter3(ax3D, score_explore(is_T2_only,1), score_explore(is_T2_only,2), score_explore(is_T2_only,3), 30, P.colorOutlierT2, 's'); pca_legend_texts_3d{end+1} = 'T2-only Outlier'; end
    if any(is_Q_only), h_pca_plots_3d(end+1) = scatter3(ax3D, score_explore(is_Q_only,1), score_explore(is_Q_only,2), score_explore(is_Q_only,3), 30, P.colorOutlierQ, 'd'); pca_legend_texts_3d{end+1} = 'Q-only Outlier'; end
    if any(is_T2_and_Q), h_pca_plots_3d(end+1) = scatter3(ax3D, score_explore(is_T2_and_Q,1), score_explore(is_T2_and_Q,2), score_explore(is_T2_and_Q,3), 35, P.colorOutlierBoth, '*'); pca_legend_texts_3d{end+1} = 'T2 & Q Outlier'; end
    hold(ax3D, 'off'); view(ax3D, -30, 20);
    xlabel(ax3D, sprintf('PC1 (%.2f%%)', explained_explore(1)));
    ylabel(ax3D, sprintf('PC2 (%.2f%%)', explained_explore(2)));
    zlabel(ax3D, sprintf('PC3 (%.2f%%)', explained_explore(3)));
    title(ax3D, '3D PCA Score Plot (PC1-3) - Exploratory', 'FontWeight','normal');
    if ~isempty(h_pca_plots_3d), legend(h_pca_plots_3d, pca_legend_texts_3d, 'Location', 'best', 'FontSize', P.plotFontSize-1); end
    grid(ax3D, 'on'); axis(ax3D,'tight'); set(gca, 'FontSize', P.plotFontSize);
    exportgraphics(fig_pca_3d, fullfile(figuresDir, sprintf('%s_Exploratory_PCA_Scores_3D.tiff', P.datePrefix)), 'Resolution', 300);
    savefig(fig_pca_3d, fullfile(figuresDir, sprintf('%s_Exploratory_PCA_Scores_3D.fig', P.datePrefix)));
end

% Plot 3.6: Combined PCA Loadings Plot for k_model_explore PCs
num_pcs_to_plot_loadings = k_model_explore; 

if num_pcs_to_plot_loadings > 0 && ~isempty(coeff_explore)
    
    % Determine a reasonable tiled layout (e.g., aim for max 2-3 columns)
    if num_pcs_to_plot_loadings <= 3
        ncols_loadings = 1;
        nrows_loadings = num_pcs_to_plot_loadings;
    elseif num_pcs_to_plot_loadings <= 8 % Max 2 columns for up to 8 PCs
        ncols_loadings = 2;
        nrows_loadings = ceil(num_pcs_to_plot_loadings / ncols_loadings);
    else % Max 3 columns for more than 8 PCs
        ncols_loadings = 3;
        nrows_loadings = ceil(num_pcs_to_plot_loadings / ncols_loadings);
    end

    fig_loadings_combined = figure('Name', sprintf('Exploratory: PCA Loadings (PC1 to PC%d)', num_pcs_to_plot_loadings));
    fig_loadings_combined.Position = [100, 100, min(ncols_loadings * 400, 1200), min(nrows_loadings * 250, 900)]; % Adjust size
    
    tl_loadings = tiledlayout(nrows_loadings, ncols_loadings, 'TileSpacing','compact', 'Padding', 'compact');
    title(tl_loadings, sprintf('PCA Loadings for the %d PCs used in T2/Q Model Construction', num_pcs_to_plot_loadings), 'FontSize', P.plotFontSize+1);

    for pc_idx = 1:num_pcs_to_plot_loadings
        if pc_idx > size(coeff_explore,2), break; end % Safety break
        
        ax_l = nexttile(tl_loadings);
        plot(ax_l, wavenumbers_roi, coeff_explore(:, pc_idx), 'LineWidth', 1.2); 
        title(ax_l, sprintf('PC%d Loadings (Expl.Var: %.2f%%)', pc_idx, explained_explore(pc_idx)), 'FontWeight','normal', 'FontSize', P.plotFontSize-1); 
        ylabel(ax_l, 'Loading Value', 'FontSize', P.plotFontSize-1); 
        grid(ax_l,'on'); 
        set(ax_l, 'XDir','reverse', 'XLim', P.plotXLim, 'FontSize', P.plotFontSize-2);
        if mod(pc_idx-1, ncols_loadings) ~= 0 % Not the first column
            set(ax_l, 'YTickLabel',[]);
        end
        % Show X-axis labels only for the bottom row of tiles
        current_tile_row = ceil(get(ax_l,'Layout').Tile / ncols_loadings);
        if current_tile_row < nrows_loadings
             set(ax_l, 'XTickLabel',[]);
        end
    end
    xlabel(tl_loadings, P.plotXLabel, 'FontSize', P.plotFontSize);
    
    exportgraphics(fig_loadings_combined, fullfile(figuresDir, sprintf('%s_Exploratory_PCA_Loadings_Combined_kModel.tiff', P.datePrefix)), 'Resolution', 300);
    savefig(fig_loadings_combined, fullfile(figuresDir, sprintf('%s_Exploratory_PCA_Loadings_Combined_kModel.fig', P.datePrefix)));
    fprintf('Combined PCA loadings plot (k_model PCs) saved.\n');
else
    fprintf('Skipping combined PCA loadings plot as k_model_explore is 0 or coeffs are empty.\n');
end

% Plot 3.7: Tiled Layout of Outlier Spectra Categories
fprintf('Generating plot of spectra by outlier category...\n');

% Flags 'is_T2_only', 'is_Q_only', 'is_T2_and_Q' are assumed to be defined from:
% is_T2_only = flag_T2_outlier_explore & ~flag_Q_outlier_explore;
% is_Q_only = ~flag_T2_outlier_explore & flag_Q_outlier_explore;
% is_T2_and_Q = flag_AND_outlier_explore; % (which is flag_T2_outlier_explore & flag_Q_outlier_explore)

fig_outlier_spectra_cats = figure('Name', 'Exploratory: Spectra of Outlier Categories');
fig_outlier_spectra_cats.Position = [100 100 700 800]; % Adjust as needed
tl_outcat = tiledlayout(3,1, 'TileSpacing','compact', 'Padding','compact');
title(tl_outcat, 'Spectra by Outlier Category (All Training Data)', 'FontSize', P.plotFontSize+1);

% Category 1: Q-only Outliers
ax_q_only = nexttile(tl_outcat);
hold(ax_q_only, 'on');
spectra_q_only = X_train_explore(is_Q_only, :);
num_q_only = sum(is_Q_only);
if num_q_only > 0
    plot(ax_q_only, wavenumbers_roi, spectra_q_only', 'Color', [P.colorOutlierQ, 0.3]); % Translucent lines
    plot(ax_q_only, wavenumbers_roi, mean(spectra_q_only,1), 'Color', P.colorOutlierQ, 'LineWidth', 1.5, 'DisplayName', sprintf('Mean Q-only (n=%d)', num_q_only));
end
title(ax_q_only, sprintf('Q-only Flagged Spectra (n=%d)', num_q_only), 'FontWeight', 'normal', 'FontSize', P.plotFontSize);
ylabel(ax_q_only, P.plotYLabelAbsorption);
set(ax_q_only, 'XDir','reverse', 'XLim', P.plotXLim, 'XTickLabel', [], 'FontSize', P.plotFontSize-1);
grid(ax_q_only, 'on');
if num_q_only > 0, legend(ax_q_only, 'show', 'Location', 'northeastoutside', 'FontSize', P.plotFontSize-2); end
hold(ax_q_only, 'off');

% Category 2: T2-only Outliers
ax_t2_only = nexttile(tl_outcat);
hold(ax_t2_only, 'on');
spectra_t2_only = X_train_explore(is_T2_only, :);
num_t2_only = sum(is_T2_only);
if num_t2_only > 0
    plot(ax_t2_only, wavenumbers_roi, spectra_t2_only', 'Color', [P.colorOutlierT2, 0.3]);
    plot(ax_t2_only, wavenumbers_roi, mean(spectra_t2_only,1), 'Color', P.colorOutlierT2, 'LineWidth', 1.5, 'DisplayName', sprintf('Mean T2-only (n=%d)', num_t2_only));
end
title(ax_t2_only, sprintf('T2-only Flagged Spectra (n=%d)', num_t2_only), 'FontWeight', 'normal', 'FontSize', P.plotFontSize);
ylabel(ax_t2_only, P.plotYLabelAbsorption);
set(ax_t2_only, 'XDir','reverse', 'XLim', P.plotXLim, 'XTickLabel', [], 'FontSize', P.plotFontSize-1);
grid(ax_t2_only, 'on');
if num_t2_only > 0, legend(ax_t2_only, 'show', 'Location', 'northeastoutside', 'FontSize', P.plotFontSize-2); end
hold(ax_t2_only, 'off');

% Category 3: T2 & Q (Consensus) Outliers
ax_t2_and_q = nexttile(tl_outcat);
hold(ax_t2_and_q, 'on');
spectra_t2_and_q = X_train_explore(is_T2_and_Q, :);
num_t2_and_q = sum(is_T2_and_Q);
if num_t2_and_q > 0
    plot(ax_t2_and_q, wavenumbers_roi, spectra_t2_and_q', 'Color', [P.colorOutlierBoth, 0.3]);
    plot(ax_t2_and_q, wavenumbers_roi, mean(spectra_t2_and_q,1), 'Color', P.colorOutlierBoth, 'LineWidth', 1.5, 'DisplayName', sprintf('Mean T2&Q (n=%d)', num_t2_and_q));
end
title(ax_t2_and_q, sprintf('T2 & Q Flagged Spectra (Consensus, n=%d)', num_t2_and_q), 'FontWeight', 'normal', 'FontSize', P.plotFontSize);
ylabel(ax_t2_and_q, P.plotYLabelAbsorption);
xlabel(ax_t2_and_q, P.plotXLabel); % X-label only on the bottom plot
set(ax_t2_and_q, 'XDir','reverse', 'XLim', P.plotXLim, 'FontSize', P.plotFontSize-1);
grid(ax_t2_and_q, 'on');
if num_t2_and_q > 0, legend(ax_t2_and_q, 'show', 'Location', 'northeastoutside', 'FontSize', P.plotFontSize-2); end
hold(ax_t2_and_q, 'off');

exportgraphics(fig_outlier_spectra_cats, fullfile(figuresDir, sprintf('%s_Exploratory_OutlierCategory_Spectra.tiff', P.datePrefix)), 'Resolution', 300);
savefig(fig_outlier_spectra_cats, fullfile(figuresDir, sprintf('%s_Exploratory_OutlierCategory_Spectra.fig', P.datePrefix)));
fprintf('Plot of spectra by outlier category saved.\n');

% Plot 3.9: Spectra Flagged by Different Alpha Levels (using OR logic)
fprintf('Generating plot of spectra sensitive to alpha level changes...\n');

P.alpha_primary = 0.01; % Your current, stricter alpha
P.alpha_lenient = 0.05; % A more lenient alpha to compare against

% --- Recalculate Thresholds for Lenient Alpha ---
% Variables T2_values_explore, Q_values_explore, k_model_explore, n_samples_explore,
% latent_explore, num_total_pcs_latent, theta1, theta2, theta3, h0 are assumed 
% to be available from Section 2 calculations.

T2_threshold_lenient = NaN;
Q_threshold_lenient = NaN;

if n_samples_explore > k_model_explore && k_model_explore > 0
    T2_threshold_lenient = ((k_model_explore * (n_samples_explore - 1)) / (n_samples_explore - k_model_explore)) ...
        * finv(1 - P.alpha_lenient, k_model_explore, n_samples_explore - k_model_explore);
else
    T2_threshold_lenient = chi2inv(1 - P.alpha_lenient, k_model_explore);
end

if k_model_explore < num_total_pcs_latent && theta1 > eps && theta2 > eps % Check theta1,theta2 for safety
    ca_lenient = norminv(1 - P.alpha_lenient);
    val_in_bracket_lenient = ca_lenient * sqrt(2*theta2*h0^2)/theta1 + 1 + theta2*h0*(h0-1)/(theta1^2);
    if val_in_bracket_lenient > 0 && h0 > eps % ensure h0 is positive
        Q_threshold_lenient = theta1 * (val_in_bracket_lenient)^(1/h0);
    end
end
if isnan(Q_threshold_lenient) || isinf(Q_threshold_lenient)
    Q_threshold_lenient = prctile(Q_values_explore, (1-P.alpha_lenient)*100);
end

fprintf('  Primary Alpha (%.3f): T2_thresh=%.4f, Q_thresh=%.4g\n', P.alpha_primary, T2_threshold_explore, Q_threshold_explore);
fprintf('  Lenient Alpha (%.3f): T2_thresh=%.4f, Q_thresh=%.4g\n', P.alpha_lenient, T2_threshold_lenient, Q_threshold_lenient);

% --- Identify Outlier Flags for Both Alphas (using OR logic) ---
flag_T2_primary = (T2_values_explore > T2_threshold_explore);
flag_Q_primary  = (Q_values_explore > Q_threshold_explore);
flag_OR_outlier_primary = flag_T2_primary | flag_Q_primary;

flag_T2_lenient = (T2_values_explore > T2_threshold_lenient);
flag_Q_lenient  = (Q_values_explore > Q_threshold_lenient);
flag_OR_outlier_lenient = flag_T2_lenient | flag_Q_lenient;

% Spectra that are outliers with lenient alpha BUT NOT with primary alpha
alpha_sensitive_outliers_flag = flag_OR_outlier_lenient & ~flag_OR_outlier_primary;
num_alpha_sensitive = sum(alpha_sensitive_outliers_flag);

fprintf('  Number of spectra flagged by OR logic with lenient alpha (%.3f): %d\n', P.alpha_lenient, sum(flag_OR_outlier_lenient));
fprintf('  Number of spectra flagged by OR logic with primary alpha (%.3f): %d\n', P.alpha_primary, sum(flag_OR_outlier_primary));
fprintf('  Number of "alpha-sensitive" spectra (outlier at %.3f but not at %.3f): %d\n', P.alpha_lenient, P.alpha_primary, num_alpha_sensitive);

% --- Plot these alpha-sensitive spectra ---
fig_alpha_sens = figure('Name', sprintf('Exploratory: Alpha-Sensitive Outlier Spectra (%.2f vs %.2f)', P.alpha_lenient, P.alpha_primary));
fig_alpha_sens.Position = [150 150 800 600];

if num_alpha_sensitive > 0
    spectra_alpha_sensitive = X_train_explore(alpha_sensitive_outliers_flag, :);
    
    % Get corresponding WHO grades for coloring
    y_numeric_alpha_sensitive = y_train_numeric_explore(alpha_sensitive_outliers_flag);
    
    hold on;
    % Plot WHO-1 alpha-sensitive outliers
    idx_who1_sens = (y_numeric_alpha_sensitive == 1);
    if any(idx_who1_sens)
        plot(wavenumbers_roi, spectra_alpha_sensitive(idx_who1_sens, :)', 'Color', [P.colorWHO1, 0.3]);
        % Plot one non-transparent for legend
        h_sens1 = plot(wavenumbers_roi, spectra_alpha_sensitive(find(idx_who1_sens,1), :), 'Color', P.colorWHO1, 'LineWidth', 1.5, 'DisplayName', sprintf('WHO-1 (Sensitive, n=%d)', sum(idx_who1_sens)));
    end
    
    % Plot WHO-3 alpha-sensitive outliers
    idx_who3_sens = (y_numeric_alpha_sensitive == 3);
    if any(idx_who3_sens)
        plot(wavenumbers_roi, spectra_alpha_sensitive(idx_who3_sens, :)', 'Color', [P.colorWHO3, 0.3]);
        h_sens3 = plot(wavenumbers_roi, spectra_alpha_sensitive(find(idx_who3_sens,1), :), 'Color', P.colorWHO3, 'LineWidth', 1.5, 'DisplayName', sprintf('WHO-3 (Sensitive, n=%d)', sum(idx_who3_sens)));
    end
    hold off;
    
    title(sprintf('Spectra Flagged by OR Logic at alpha=%.2f but NOT at alpha=%.2f (n=%d)', P.alpha_lenient, P.alpha_primary, num_alpha_sensitive),'FontWeight','normal');
    xlabel(P.plotXLabel);
    ylabel(P.plotYLabelAbsorption);
    set(gca, 'XDir','reverse', 'XLim', P.plotXLim, 'FontSize', P.plotFontSize);
    grid on;
    
    legend_handles_alpha_sens = [];
    if any(idx_who1_sens) && exist('h_sens1','var'), legend_handles_alpha_sens = [legend_handles_alpha_sens, h_sens1]; end
    if any(idx_who3_sens) && exist('h_sens3','var'), legend_handles_alpha_sens = [legend_handles_alpha_sens, h_sens3]; end
    if ~isempty(legend_handles_alpha_sens)
        legend(legend_handles_alpha_sens, 'Location', 'best');
    end
    
else
    text(0.5, 0.5, sprintf('No spectra uniquely flagged by alpha=%.2f vs. alpha=%.2f', P.alpha_lenient, P.alpha_primary), 'Parent', gca, 'HorizontalAlignment','center');
    title(sprintf('No Alpha-Sensitive Spectra Found (%.2f vs %.2f)', P.alpha_lenient, P.alpha_primary),'FontWeight','normal');
end

exportgraphics(fig_alpha_sens, fullfile(figuresDir, sprintf('%s_Exploratory_AlphaSensitive_Spectra_%.2fvs%.2f.tiff', P.datePrefix, P.alpha_lenient, P.alpha_primary)), 'Resolution', 300);
savefig(fig_alpha_sens, fullfile(figuresDir, sprintf('%s_Exploratory_AlphaSensitive_Spectra_%.2fvs%.2f.fig', P.datePrefix, P.alpha_lenient, P.alpha_primary)));
fprintf('Plot of alpha-sensitive spectra saved.\n');




fprintf('Diagnostic visualizations generated and saved.\n');
fprintf('Please review figures in: %s\n', figuresDir);

%% --- 4. Save Exploratory Analysis Data ---
% (Code for saving `exploratoryOutlierData` as in previous response, no changes needed here)
% This saves T2_values_explore, Q_values_explore, thresholds, all flags, PCA model etc.
fprintf('\n--- 4. Saving Exploratory Analysis Data (T2/Q values, flags, PCA model) ---\n');
exploratoryOutlierData = struct();
exploratoryOutlierData.scriptRunDate = P.datePrefix;
exploratoryOutlierData.alpha_significance = P.alpha_significance;
exploratoryOutlierData.variance_to_explain_for_PCA_model = P.variance_to_explain_for_PCA_model;
exploratoryOutlierData.k_model_explore = k_model_explore;
exploratoryOutlierData.T2_values = T2_values_explore;
exploratoryOutlierData.T2_threshold = T2_threshold_explore;
exploratoryOutlierData.flag_T2_outlier = flag_T2_outlier_explore;
exploratoryOutlierData.Q_values = Q_values_explore;
exploratoryOutlierData.Q_threshold = Q_threshold_explore;
exploratoryOutlierData.flag_Q_outlier = flag_Q_outlier_explore;
exploratoryOutlierData.flag_AND_outlier = flag_AND_outlier_explore;
exploratoryOutlierData.flag_OR_outlier = flag_OR_outlier_explore;
exploratoryOutlierData.Original_ProbeRowIndices = Original_ProbeRowIndices_explore;
exploratoryOutlierData.Original_SpectrumIndexInProbe = Original_SpectrumIndexInProbe_explore;
exploratoryOutlierData.Patient_ID_explore = Patient_ID_explore;
exploratoryOutlierData.y_train_numeric_explore = y_train_numeric_explore;
exploratoryOutlierData.y_train_cat_explore = y_train_cat_explore;
exploratoryOutlierData.PCA_coeff = coeff_explore;
exploratoryOutlierData.PCA_mu = mu_explore;
exploratoryOutlierData.PCA_latent = latent_explore;
exploratoryOutlierData.PCA_explained = explained_explore;
exploratoryOutlierData.PCA_score = score_explore; 
exploratoryFilename_mat = fullfile(resultsDir, sprintf('%s_ExploratoryOutlier_AnalysisData.mat', P.datePrefix));
save(exploratoryFilename_mat, 'exploratoryOutlierData', '-v7.3');
fprintf('Exploratory outlier analysis data saved to: %s\n', exploratoryFilename_mat);

fprintf('\n--- Exploratory Outlier Analysis Script Finished ---\n');
fprintf('Review generated plots and data. Next step: Decide on removal strategy and apply in a separate script or function.\n');

% Helper function for Plot 3.7 (Adaptation of your Plot 6 logic)
% function plot_outlier_categories(ax, X_data, class_selector, normal_flag, t2_only_flag, q_only_flag, both_t2q_flag, wavenumbers, plot_params, className)
%     hold(ax, 'on');
%     mean_normal_spec = mean(X_data(class_selector & normal_flag, :), 1, 'omitnan');
%     legend_handles = []; legend_texts = {};
% 
%     if any(~isnan(mean_normal_spec))
%         legend_handles(end+1) = plot(ax, wavenumbers, mean_normal_spec, 'Color', [0 0 0], 'LineWidth', 1.5);
%         legend_texts{end+1} = sprintf('Mean %s (Normal)', className);
%     end
% 
%     cat_flags = {t2_only_flag, q_only_flag, both_t2q_flag};
%     cat_names = {'T2-only', 'Q-only', 'T2&Q'};
%     cat_colors = {plot_params.colorOutlierT2, plot_params.colorOutlierQ, plot_params.colorOutlierBoth};
% 
%     for c_idx = 1:length(cat_flags)
%         current_cat_flag = class_selector & cat_flags{c_idx};
%         spectra_this_cat = X_data(current_cat_flag, :);
%         if ~isempty(spectra_this_cat)
%             for k=1:size(spectra_this_cat,1)
%                 plot(ax, wavenumbers, spectra_this_cat(k,:), 'Color', [cat_colors{c_idx}, 0.3], 'LineWidth', 0.5, 'HandleVisibility', 'off');
%             end
%             if size(spectra_this_cat,1) > 0 % Add one representative for legend
%                 legend_handles(end+1) = plot(ax, wavenumbers, spectra_this_cat(1,:), 'Color', [cat_colors{c_idx}, 0.8], 'LineWidth', 0.7);
%                 legend_texts{end+1} = sprintf('%s: %s (n=%d)',className, cat_names{c_idx}, sum(current_cat_flag));
%             end
%         end
%     end
%     hold(ax, 'off'); title(ax, sprintf('%s: Mean Normal vs. Outlier Categories', className), 'FontWeight','normal');
%     xlabel(ax, plot_params.plotXLabel); ylabel(ax, plot_params.plotYLabelAbsorption);
%     xlim(ax, plot_params.plotXLim); ylim(ax, 'auto'); 
%     if ~isempty(legend_handles), legend(ax, legend_handles, legend_texts, 'Location', 'best', 'FontSize', plot_params.plotFontSize-2); end
%     ax.XDir = 'reverse'; grid(ax, 'on');
% end