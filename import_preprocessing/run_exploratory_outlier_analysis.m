% run_exploratory_outlier_visualization.m
%
% PURPOSE:
%   To perform an exploratory analysis of potential outliers in the 
%   training dataset using PCA, Hotelling's T-squared, and Q-residuals.
%   This script focuses on generating a specific set of visualizations 
%   to inform subsequent outlier removal decisions.
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
%   - Specific diagnostic plots (.tiff and .fig).
%   - A .mat file containing T2_values, Q_values, thresholds, and flags for
%     potential outliers, plus PCA model info and spectrum identifiers.
%
% DATE: 2025-05-17

%% --- 0. Configuration & Setup ---
clearvars -except dataTableTrain wavenumbers_roi; % Clear most, but keep inputs if already in workspace
close all;
fprintf('Starting Exploratory Outlier Visualization - %s\n', string(datetime('now')));

% --- Define Paths (User to verify/modify) ---
projectBasePath = 'C:\Users\Franz\OneDrive\01_Promotion\01 Data\meningioma-ftir-classification'; 
dataDir    = fullfile(projectBasePath, 'data');
resultsDir = fullfile(projectBasePath, 'results', 'Phase1_ExploratoryOutlierVis'); % New specific subfolder
figuresDir = fullfile(projectBasePath, 'figures', 'Phase1_ExploratoryOutlierVis'); % New specific subfolder

if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
if ~exist(figuresDir, 'dir'), mkdir(figuresDir); end

% --- Parameters ---
P.alpha_significance = 0.01; % Primary alpha for threshold calculation
P.variance_to_explain_for_PCA_model = 0.95; % For k_model determination
P.datePrefix = string(datetime('now','Format','yyyyMMdd'));

% --- Plotting Defaults ---
P.colorWHO1 = [0.9, 0.6, 0.4]; 
P.colorWHO3 = [0.4, 0.702, 0.902]; 
P.colorOutlierGeneric = [0.7 0.7 0.7]; % Generic color for points marked as "outlier" in some plots
P.colorT2Outlier = [0.8, 0.2, 0.2];   
P.colorQOutlier = [0.2, 0.2, 0.8];    
P.colorBothOutlier = [0.8, 0, 0.8]; 

P.plotFontSize = 10;
P.plotXLabel = 'Wellenzahl (cm^{-1})';
P.plotYLabelAbsorption = 'Absorption (a.u.)';
P.plotXLim = [950 1800];

fprintf('Setup complete. Results will be saved in: \n  %s\n  %s\n', resultsDir, figuresDir);

%% --- 1. Load and Prepare Training Data ---
fprintf('\n--- 1. Loading and Preparing Training Data ---\n');

% Attempt to load dataTableTrain if not already in workspace
if ~exist('dataTableTrain', 'var')
    trainDataTableFile = fullfile(dataDir, 'data_table_train.mat'); 
    if exist(trainDataTableFile, 'file')
        fprintf('Loading dataTableTrain from: %s\n', trainDataTableFile);
        loadedVars = load(trainDataTableFile);
        if isfield(loadedVars, 'dataTableTrain')
            dataTableTrain = loadedVars.dataTableTrain;
        else
            error('Variable "dataTableTrain" not found within %s.', trainDataTableFile);
        end
    else
        error('Input file %s not found AND dataTableTrain not in workspace.', trainDataTableFile);
    end
end
fprintf('dataTableTrain available with %d probes.\n', height(dataTableTrain));

% Attempt to load wavenumbers_roi if not already in workspace
if ~exist('wavenumbers_roi', 'var')
    try
        load(fullfile(dataDir, 'wavenumbers.mat'), 'wavenumbers_roi');
        if iscolumn(wavenumbers_roi), wavenumbers_roi = wavenumbers_roi'; end
    catch ME_wave
        error('Error loading wavenumbers.mat: %s', ME_wave.message);
    end
end
fprintf('wavenumbers_roi available (%d points).\n', length(wavenumbers_roi));

% --- Flatten dataTableTrain into X_train_explore, y_train_numeric_explore, etc. ---
allSpectra_cell = {}; allLabels_cell = {}; allPatientIDs_cell = {};
allOriginalProbeRowIndices_vec = []; allOriginalSpectrumIndices_InProbe_vec = []; 
fprintf('Extracting and flattening spectra from dataTableTrain.CombinedSpectra...\n');
for i = 1:height(dataTableTrain)
    spectraMatrix = dataTableTrain.CombinedSpectra{i}; 
    if isempty(spectraMatrix) || ~isnumeric(spectraMatrix) || ndims(spectraMatrix) ~= 2, warning('Probe %s (Row %d): CombinedSpectra invalid. Skipping.', dataTableTrain.Diss_ID{i}, i); continue; end
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
if isempty(allSpectra_cell), error('No valid spectra extracted from dataTableTrain.CombinedSpectra.'); end

X_explore = cell2mat(allSpectra_cell);
y_cat_explore = cat(1, allLabels_cell{:});
Patient_ID_explore = vertcat(allPatientIDs_cell{:}); 
Original_ProbeRowIndices_explore = allOriginalProbeRowIndices_vec; 
Original_SpectrumIndexInProbe_explore = allOriginalSpectrumIndices_InProbe_vec;

y_numeric_explore = zeros(length(y_cat_explore), 1);
categories_y = categories(y_cat_explore);
idx_who1 = find(strcmp(categories_y, 'WHO-1')); idx_who3 = find(strcmp(categories_y, 'WHO-3'));
if ~isempty(idx_who1), y_numeric_explore(y_cat_explore == categories_y{idx_who1}) = 1; end
if ~isempty(idx_who3), y_numeric_explore(y_cat_explore == categories_y{idx_who3}) = 3; end
fprintf('Data for exploration: %d spectra, %d features.\n', size(X_explore,1), size(X_explore,2));

%% --- 2. Perform PCA, Calculate T² and Q Statistics & Thresholds ---
fprintf('\n--- 2. PCA, T2/Q Calculation & Thresholds ---\n');
[coeff_explore, score_explore, latent_explore, ~, explained_explore, mu_explore] = ...
    pca(X_explore, 'Algorithm','svd'); % Removed tsquared_builtin as we calculate T2 manually
fprintf('PCA completed. Variance by 1st PC: %.2f%%\n', explained_explore(1));

cumulativeVariance = cumsum(explained_explore);
k_model_explore = find(cumulativeVariance >= P.variance_to_explain_for_PCA_model*100, 1, 'first');
if isempty(k_model_explore), k_model_explore = min(length(explained_explore), size(X_explore,1)-1); end
if k_model_explore == 0 && ~isempty(explained_explore), k_model_explore = 1; end
if isempty(explained_explore) || k_model_explore == 0, error('Could not determine k_model for T2/Q.'); end
fprintf('k_model (for T2/Q) based on >=%.0f%% variance: %d PCs.\n', P.variance_to_explain_for_PCA_model*100, k_model_explore);

% T² Calculation
score_k_model = score_explore(:, 1:k_model_explore);
lambda_k_model = latent_explore(1:k_model_explore); 
lambda_k_model(lambda_k_model <= eps) = eps; % Avoid division by zero or negative
T2_values = sum(bsxfun(@rdivide, score_k_model.^2, lambda_k_model'), 2);

% T² Threshold
n_samples = size(X_explore, 1);
if n_samples > k_model_explore && k_model_explore > 0
    T2_threshold = ((k_model_explore * (n_samples - 1)) / (n_samples - k_model_explore)) ...
        * finv(1 - P.alpha_significance, k_model_explore, n_samples - k_model_explore);
else
    T2_threshold = chi2inv(1 - P.alpha_significance, k_model_explore); % Fallback
end
fprintf('Hotelling T2 threshold (alpha=%.2f): %.4f\n', P.alpha_significance, T2_threshold);

% Q-Statistic (SPE) Calculation
X_reconstructed = score_explore(:, 1:k_model_explore) * coeff_explore(:, 1:k_model_explore)' + mu_explore;
E_residuals = X_explore - X_reconstructed;
Q_values = sum(E_residuals.^2, 2);

% Q-Statistic Threshold (Jackson-Mudholkar)
num_total_pcs_latent = length(latent_explore); % Should be rank of X_explore
Q_threshold = NaN;
if k_model_explore < num_total_pcs_latent
    discarded_eigenvalues = latent_explore(k_model_explore+1:num_total_pcs_latent); % Ensure end index is valid
    discarded_eigenvalues(discarded_eigenvalues <= eps) = eps;
    theta1 = sum(discarded_eigenvalues);
    theta2 = sum(discarded_eigenvalues.^2);
    theta3 = sum(discarded_eigenvalues.^3);
    if theta1 > eps && theta2 > eps % Check to prevent division by zero or issues with h0
        h0 = 1 - (2 * theta1 * theta3) / (3 * theta2^2);
        if h0 <= eps, h0 = 1; fprintf('Warning: Q-thresh h0 was <=0 or invalid, set to 1 for stability.\n'); end
        ca = norminv(1 - P.alpha_significance);
        val_in_bracket = ca * sqrt(2*theta2*h0^2)/theta1 + 1 + theta2*h0*(h0-1)/(theta1^2);
        if val_in_bracket > 0
            Q_threshold = theta1 * (val_in_bracket)^(1/h0);
        end
    elseif theta1 > eps % Simpler case if theta2 is effectively zero
        Q_threshold = theta1; % A very rough approximation if higher moments are zero
         fprintf('Warning: Q-thresh theta2 near zero, using simplified Q_threshold approx.\n');
    end
end
if isnan(Q_threshold) || isinf(Q_threshold) % Fallback if theoretical calculation failed
    Q_threshold = prctile(Q_values, (1-P.alpha_significance)*100);
    fprintf('Using empirical Q-threshold (%.2fth percentile): %.4g\n', (1-P.alpha_significance)*100, Q_threshold);
else
    fprintf('Q-Statistic (SPE) threshold (alpha=%.2f): %.4g\n', P.alpha_significance, Q_threshold);
end

% --- Define Outlier Flags based on calculated thresholds ---
flag_T2_outlier = (T2_values > T2_threshold);
flag_Q_outlier  = (Q_values > Q_threshold);

% Specific categories for plotting
is_T2_only = flag_T2_outlier & ~flag_Q_outlier;
is_Q_only  = ~flag_T2_outlier & flag_Q_outlier;
is_T2_and_Q = flag_T2_outlier & flag_Q_outlier; % Consensus outliers
is_normal = ~flag_T2_outlier & ~flag_Q_outlier;   % Not flagged by either

fprintf('Spectra counts by category: Normal=%d, T2-only=%d, Q-only=%d, T2&Q=%d\n', ...
    sum(is_normal), sum(is_T2_only), sum(is_Q_only), sum(is_T2_and_Q));

%% --- 3. Generate Requested Visualizations ---
fprintf('\n--- 3. Generating Requested Visualizations ---\n');

% Plot 1: Q-Statistic and T2-Statistic Individual Plots (in a tiled layout)
fig1_tq_individual = figure('Name', 'Exploratory: Individual T2 and Q Statistics');
fig1_tq_individual.Position = [50, 500, 900, 600];
tl_tq = tiledlayout(2,1, 'TileSpacing','compact', 'Padding','compact');
title(tl_tq, 'Individual T2 and Q Statistic Distributions', 'FontSize', P.plotFontSize+2);

% Subplot for T2
ax_t2 = nexttile(tl_tq);
plot(ax_t2, 1:n_samples, T2_values, 'o', 'Color', [0.5 0.5 0.5], 'MarkerSize', 3, 'DisplayName', 'T2 Values'); hold(ax_t2, 'on');
plot(ax_t2, find(flag_T2_outlier), T2_values(flag_T2_outlier), 'x', 'Color', P.colorT2Outlier, 'MarkerSize', 5, 'DisplayName', 'T2 > Threshold');
yline(ax_t2, T2_threshold, '--', 'Color', P.colorT2Outlier, 'LineWidth', 1.5, 'DisplayName', sprintf('T2 Thresh (%.0f%%)',(1-P.alpha_significance)*100));
hold(ax_t2, 'off'); xlabel(ax_t2, 'Spectrum Index'); ylabel(ax_t2, 'Hotelling T^2 Value');
title(ax_t2, sprintf('Hotelling T^2 Values (k_{model}=%d PCs)', k_model_explore), 'FontWeight','normal');
legend(ax_t2, 'show', 'Location', 'northeast', 'FontSize', P.plotFontSize-1); grid(ax_t2, 'on'); set(ax_t2, 'FontSize', P.plotFontSize-1);

% Subplot for Q
ax_q = nexttile(tl_tq);
plot(ax_q, 1:n_samples, Q_values, 'o', 'Color', [0.5 0.5 0.5], 'MarkerSize', 3, 'DisplayName', 'Q Values'); hold(ax_q, 'on');
plot(ax_q, find(flag_Q_outlier), Q_values(flag_Q_outlier), 'x', 'Color', P.colorQOutlier, 'MarkerSize', 5, 'DisplayName', 'Q > Threshold');
yline(ax_q, Q_threshold, '--', 'Color', P.colorQOutlier, 'LineWidth', 1.5, 'DisplayName', sprintf('Q Thresh (%.0f%%)',(1-P.alpha_significance)*100));
hold(ax_q, 'off'); xlabel(ax_q, 'Spectrum Index'); ylabel(ax_q, 'Q-Statistic (SPE)');
title(ax_q, sprintf('Q-Statistic (SPE) Values (k_{model}=%d PCs)', k_model_explore), 'FontWeight','normal');
legend(ax_q, 'show', 'Location', 'northeast', 'FontSize', P.plotFontSize-1); grid(ax_q, 'on'); set(ax_q, 'FontSize', P.plotFontSize-1);

exportgraphics(fig1_tq_individual, fullfile(figuresDir, sprintf('%s_Vis1_T2_Q_Individual.tiff', P.datePrefix)), 'Resolution', 300);
savefig(fig1_tq_individual, fullfile(figuresDir, sprintf('%s_Vis1_T2_Q_Individual.fig', P.datePrefix)));
fprintf('Plot 1 (Individual T2 & Q) saved.\n');


% Plot 2: T2 vs Q Plot with Thresholds and Outlier Categories
fig2_t2q_categories = figure('Name', 'Exploratory: T2 vs Q Plot with Categories');
fig2_t2q_categories.Position = [100 100 750 600];
hold on;
h_plots_t2q_cat = []; legend_texts_t2q_cat = {};
if any(is_normal & y_numeric_explore==1), h_plots_t2q_cat(end+1) = plot(T2_values(is_normal & y_numeric_explore==1), Q_values(is_normal & y_numeric_explore==1), 'o', 'Color', P.colorWHO1, 'MarkerSize', 4, 'MarkerFaceColor', P.colorWHO1); legend_texts_t2q_cat{end+1} = 'WHO-1 (Normal)'; end
if any(is_normal & y_numeric_explore==3), h_plots_t2q_cat(end+1) = plot(T2_values(is_normal & y_numeric_explore==3), Q_values(is_normal & y_numeric_explore==3), 'o', 'Color', P.colorWHO3, 'MarkerSize', 4, 'MarkerFaceColor', P.colorWHO3); legend_texts_t2q_cat{end+1} = 'WHO-3 (Normal)'; end
if any(is_T2_only), h_plots_t2q_cat(end+1) = plot(T2_values(is_T2_only), Q_values(is_T2_only), 's', 'Color', P.colorT2Outlier, 'MarkerSize', 5); legend_texts_t2q_cat{end+1} = 'T2-only Outlier'; end
if any(is_Q_only), h_plots_t2q_cat(end+1) = plot(T2_values(is_Q_only), Q_values(is_Q_only), 'd', 'Color', P.colorQOutlier, 'MarkerSize', 5); legend_texts_t2q_cat{end+1} = 'Q-only Outlier'; end
if any(is_T2_and_Q), h_plots_t2q_cat(end+1) = plot(T2_values(is_T2_and_Q), Q_values(is_T2_and_Q), '*', 'Color', P.colorBothOutlier, 'MarkerSize', 6); legend_texts_t2q_cat{end+1} = 'T2 & Q Outlier'; end
line([T2_threshold, T2_threshold], get(gca,'YLim'), 'Color', P.colorT2Outlier, 'LineStyle', '--', 'LineWidth', 1.2, 'HandleVisibility','off');
line(get(gca,'XLim'), [Q_threshold, Q_threshold], 'Color', P.colorQOutlier, 'LineStyle', '--', 'LineWidth', 1.2, 'HandleVisibility','off');
text(T2_threshold, get(gca,'YLim')*[0.05; 0.95], sprintf(' T2 Thresh (%.2f)',T2_threshold), 'Color', P.colorT2Outlier, 'VerticalAlignment','bottom', 'HorizontalAlignment','left','FontSize',P.plotFontSize-2);
text(get(gca,'XLim')*[0.95; 0.05], Q_threshold, sprintf(' Q Thresh (%.2g)',Q_threshold), 'Color', P.colorQOutlier, 'VerticalAlignment','bottom', 'HorizontalAlignment','right','FontSize',P.plotFontSize-2);
hold off;
xlabel(sprintf('Hotelling T^2 Value')); 
ylabel(sprintf('Q-Statistic (SPE) Value'));
title(sprintf('T^2 vs. Q Plot with Outlier Categories (k_{model}=%d PCs)', k_model_explore), 'FontWeight','normal');
if ~isempty(h_plots_t2q_cat), legend(h_plots_t2q_cat, legend_texts_t2q_cat, 'Location', 'NorthEast', 'FontSize', P.plotFontSize-1); end
grid on; set(gca, 'FontSize', P.plotFontSize);
xlim_max_t2q_cat = max(T2_threshold*1.5, prctile(T2_values, 99.9)); ylim_max_t2q_cat = max(Q_threshold*1.5, prctile(Q_values, 99.9));
if xlim_max_t2q_cat > 0, xlim([0 xlim_max_t2q_cat]); else xlim auto; end; 
if ylim_max_t2q_cat > 0, ylim([0 ylim_max_t2q_cat]); else ylim auto; end;
exportgraphics(fig2_t2q_categories, fullfile(figuresDir, sprintf('%s_Vis2_T2_vs_Q_Categories.tiff', P.datePrefix)), 'Resolution', 300);
savefig(fig2_t2q_categories, fullfile(figuresDir, sprintf('%s_Vis2_T2_vs_Q_Categories.fig', P.datePrefix)));
fprintf('Plot 2 (T2 vs Q with Categories & Thresholds) saved.\n');


% Plot 3: T2 vs Q Plot (Only WHO-1 and WHO-3, no outlier marking, no thresholds)
fig3_t2q_raw_who = figure('Name', 'Exploratory: T2 vs Q Raw Distribution by WHO Grade');
fig3_t2q_raw_who.Position = [150 150 750 600];
hold on;
h_plots_t2q_raw = []; legend_texts_t2q_raw = {};
idx_who1_all = (y_numeric_explore == 1);
if any(idx_who1_all), h_plots_t2q_raw(end+1) = scatter(T2_values(idx_who1_all), Q_values(idx_who1_all), 15, P.colorWHO1, 'o', 'filled', 'MarkerFaceAlpha', 0.3); legend_texts_t2q_raw{end+1} = 'WHO-1 (All Spectra)'; end
idx_who3_all = (y_numeric_explore == 3);
if any(idx_who3_all), h_plots_t2q_raw(end+1) = scatter(T2_values(idx_who3_all), Q_values(idx_who3_all), 15, P.colorWHO3, 's', 'filled', 'MarkerFaceAlpha', 0.3); legend_texts_t2q_raw{end+1} = 'WHO-3 (All Spectra)'; end
idx_unknown_all = ~(idx_who1_all | idx_who3_all);
if any(idx_unknown_all), h_plots_t2q_raw(end+1) = scatter(T2_values(idx_unknown_all), Q_values(idx_unknown_all),15, [0.5 0.5 0.5], '^', 'filled', 'MarkerFaceAlpha', 0.2); legend_texts_t2q_raw{end+1} = 'Unknown Grade'; end
hold off; xlabel('Hotelling T^2 Value'); ylabel('Q-Statistic (SPE) Value');
title(sprintf('T^2 vs. Q Raw Data Distribution (k_{model}=%d PCs)', k_model_explore), 'FontWeight','normal');
if ~isempty(h_plots_t2q_raw), legend(h_plots_t2q_raw, legend_texts_t2q_raw, 'Location', 'best', 'FontSize', P.plotFontSize-1); end
grid on; set(gca, 'FontSize', P.plotFontSize);
xlim_data_raw = [min(T2_values(isfinite(T2_values))) max(T2_values(isfinite(T2_values)))]; ylim_data_raw = [min(Q_values(isfinite(Q_values))) max(Q_values(isfinite(Q_values)))];
if diff(xlim_data_raw) < eps, xlim_data_raw(2) = xlim_data_raw(1) + max(1, abs(xlim_data_raw(1)*0.1)); xlim_data_raw(1) = xlim_data_raw(1) - max(1, abs(xlim_data_raw(1)*0.1)); if xlim_data_raw(1) < 0 && xlim_data_raw(2) > 0 && xlim_data_raw(1)~=xlim_data_raw(2) ; else xlim_data_raw(1)=0; end; end 
if diff(ylim_data_raw) < eps, ylim_data_raw(2) = ylim_data_raw(1) + max(1, abs(ylim_data_raw(1)*0.1)); ylim_data_raw(1) = ylim_data_raw(1) - max(1, abs(ylim_data_raw(1)*0.1)); if ylim_data_raw(1) < 0 && ylim_data_raw(2) > 0 && ylim_data_raw(1)~=ylim_data_raw(2); else ylim_data_raw(1)=0; end; end
xlim(xlim_data_raw); ylim(ylim_data_raw);
exportgraphics(fig3_t2q_raw_who, fullfile(figuresDir, sprintf('%s_Vis3_T2_vs_Q_WHOgradesOnly.tiff', P.datePrefix)), 'Resolution', 300);
savefig(fig3_t2q_raw_who, fullfile(figuresDir, sprintf('%s_Vis3_T2_vs_Q_WHOgradesOnly.fig', P.datePrefix)));
fprintf('Plot 3 (T2 vs Q Raw WHO Grades) saved.\n');


% Plot 4: PCA Score Plots (2D and 3D, two versions each)
fig4_pca_scores_combined = figure('Name', 'Exploratory: PCA Score Plots');
fig4_pca_scores_combined.Position = [50 50 1000 800];
tl_pca = tiledlayout(2,2, 'TileSpacing','compact', 'Padding','compact');
title(tl_pca, 'PCA Score Plots - Training Data Exploration', 'FontSize', P.plotFontSize+2);

% Subplot 4.1: PC1 vs PC2, colored by WHO Grade only
ax_pca_2d_who = nexttile(tl_pca);
hold(ax_pca_2d_who, 'on');
h_pca_2d_who = []; leg_pca_2d_who = {};
if any(idx_who1_all), h_pca_2d_who(end+1) = scatter(ax_pca_2d_who, score_explore(idx_who1_all, 1), score_explore(idx_who1_all, min(2,size(score_explore,2))), 15, P.colorWHO1, 'o', 'filled', 'MarkerFaceAlpha', 0.3); leg_pca_2d_who{end+1} = 'WHO-1'; end
if any(idx_who3_all), h_pca_2d_who(end+1) = scatter(ax_pca_2d_who, score_explore(idx_who3_all, 1), score_explore(idx_who3_all, min(2,size(score_explore,2))), 15, P.colorWHO3, 's', 'filled', 'MarkerFaceAlpha', 0.3); leg_pca_2d_who{end+1} = 'WHO-3'; end
if any(idx_unknown_all), h_pca_2d_who(end+1) = scatter(ax_pca_2d_who, score_explore(idx_unknown_all, 1), score_explore(idx_unknown_all, min(2,size(score_explore,2))), 15, [0.5 0.5 0.5], '^', 'filled', 'MarkerFaceAlpha', 0.2); leg_pca_2d_who{end+1} = 'Unknown'; end
hold(ax_pca_2d_who, 'off');
xlabel(ax_pca_2d_who, sprintf('PC1 (%.1f%%)', explained_explore(1))); 
ylabel(ax_pca_2d_who, sprintf('PC2 (%.1f%%)', explained_explore(min(2,length(explained_explore)))));
title(ax_pca_2d_who, 'PC1 vs PC2 (by WHO Grade)', 'FontWeight','normal');
if ~isempty(h_pca_2d_who), legend(ax_pca_2d_who, h_pca_2d_who, leg_pca_2d_who, 'Location', 'best', 'FontSize', P.plotFontSize-2); end
axis(ax_pca_2d_who,'equal'); grid(ax_pca_2d_who,'on'); set(ax_pca_2d_who, 'FontSize', P.plotFontSize-1);

% Subplot 4.2: PC1 vs PC2, with Outliers Marked (using OR logic for "outlier")
ax_pca_2d_outlier = nexttile(tl_pca);
flag_OR_outlier = flag_T2_outlier | flag_Q_outlier; % Outlier if T2 or Q flags it
hold(ax_pca_2d_outlier, 'on');
h_pca_2d_out = []; leg_pca_2d_out = {};
is_normal_or_strat = ~flag_OR_outlier;
if any(is_normal_or_strat & idx_who1_all), h_pca_2d_out(end+1) = scatter(ax_pca_2d_outlier, score_explore(is_normal_or_strat & idx_who1_all, 1), score_explore(is_normal_or_strat & idx_who1_all, min(2,size(score_explore,2))), 15, P.colorWHO1, 'o', 'filled', 'MarkerFaceAlpha', 0.3); leg_pca_2d_out{end+1} = 'WHO-1 (Normal)'; end
if any(is_normal_or_strat & idx_who3_all), h_pca_2d_out(end+1) = scatter(ax_pca_2d_outlier, score_explore(is_normal_or_strat & idx_who3_all, 1), score_explore(is_normal_or_strat & idx_who3_all, min(2,size(score_explore,2))), 15, P.colorWHO3, 's', 'filled', 'MarkerFaceAlpha', 0.3); leg_pca_2d_out{end+1} = 'WHO-3 (Normal)'; end
if any(flag_OR_outlier), h_pca_2d_out(end+1) = scatter(ax_pca_2d_outlier, score_explore(flag_OR_outlier, 1), score_explore(flag_OR_outlier, min(2,size(score_explore,2))), 25, P.colorOutlierGeneric, 'x'); leg_pca_2d_out{end+1} = 'Outlier (T2 or Q)'; end
hold(ax_pca_2d_outlier, 'off');
xlabel(ax_pca_2d_outlier, sprintf('PC1 (%.1f%%)', explained_explore(1))); 
ylabel(ax_pca_2d_outlier, sprintf('PC2 (%.1f%%)', explained_explore(min(2,length(explained_explore)))));
title(ax_pca_2d_outlier, 'PC1 vs PC2 (Outliers Marked)', 'FontWeight','normal');
if ~isempty(h_pca_2d_out), legend(ax_pca_2d_outlier, h_pca_2d_out, leg_pca_2d_out, 'Location', 'best', 'FontSize', P.plotFontSize-2); end
axis(ax_pca_2d_outlier,'equal'); grid(ax_pca_2d_outlier,'on'); set(ax_pca_2d_outlier, 'FontSize', P.plotFontSize-1);

% Subplot 4.3: PC1-PC2-PC3 3D, colored by WHO Grade only
ax_pca_3d_who = nexttile(tl_pca);
if size(score_explore,2) >= 3
    hold(ax_pca_3d_who, 'on');
    h_pca_3d_who = []; leg_pca_3d_who = {};
    if any(idx_who1_all), h_pca_3d_who(end+1) = scatter3(ax_pca_3d_who, score_explore(idx_who1_all, 1), score_explore(idx_who1_all, 2), score_explore(idx_who1_all, 3), 15, P.colorWHO1, 'o', 'filled', 'MarkerFaceAlpha', 0.3); leg_pca_3d_who{end+1} = 'WHO-1'; end
    if any(idx_who3_all), h_pca_3d_who(end+1) = scatter3(ax_pca_3d_who, score_explore(idx_who3_all, 1), score_explore(idx_who3_all, 2), score_explore(idx_who3_all, 3), 15, P.colorWHO3, 's', 'filled', 'MarkerFaceAlpha', 0.3); leg_pca_3d_who{end+1} = 'WHO-3'; end
    if any(idx_unknown_all), h_pca_3d_who(end+1) = scatter3(ax_pca_3d_who, score_explore(idx_unknown_all, 1), score_explore(idx_unknown_all, 2), score_explore(idx_unknown_all, 3), 15, [0.5 0.5 0.5], '^', 'filled', 'MarkerFaceAlpha', 0.2); leg_pca_3d_who{end+1} = 'Unknown'; end
    hold(ax_pca_3d_who, 'off'); view(ax_pca_3d_who, -30, 20);
    xlabel(ax_pca_3d_who, sprintf('PC1 (%.1f%%)', explained_explore(1))); 
    ylabel(ax_pca_3d_who, sprintf('PC2 (%.1f%%)', explained_explore(2)));
    zlabel(ax_pca_3d_who, sprintf('PC3 (%.1f%%)', explained_explore(3)));
    title(ax_pca_3d_who, 'PC1-PC2-PC3 (by WHO Grade)', 'FontWeight','normal');
    if ~isempty(h_pca_3d_who), legend(ax_pca_3d_who, h_pca_3d_who, leg_pca_3d_who, 'Location', 'best', 'FontSize', P.plotFontSize-2); end
    grid(ax_pca_3d_who,'on'); axis(ax_pca_3d_who,'tight'); set(ax_pca_3d_who, 'FontSize', P.plotFontSize-1);
else
    text(0.5,0.5, 'Less than 3 PCs available', 'Parent', ax_pca_3d_who, 'HorizontalAlignment','center');
    title(ax_pca_3d_who, 'PC1-PC2-PC3 (by WHO Grade)', 'FontWeight','normal');
end

% Subplot 4.4: PC1-PC2-PC3 3D, with Outliers Marked
ax_pca_3d_outlier = nexttile(tl_pca);
if size(score_explore,2) >= 3
    hold(ax_pca_3d_outlier, 'on');
    h_pca_3d_out = []; leg_pca_3d_out = {};
    if any(is_normal_or_strat & idx_who1_all), h_pca_3d_out(end+1) = scatter3(ax_pca_3d_outlier, score_explore(is_normal_or_strat & idx_who1_all, 1), score_explore(is_normal_or_strat & idx_who1_all, 2), score_explore(is_normal_or_strat & idx_who1_all, 3), 15, P.colorWHO1, 'o', 'filled', 'MarkerFaceAlpha', 0.3); leg_pca_3d_out{end+1} = 'WHO-1 (Normal)'; end
    if any(is_normal_or_strat & idx_who3_all), h_pca_3d_out(end+1) = scatter3(ax_pca_3d_outlier, score_explore(is_normal_or_strat & idx_who3_all, 1), score_explore(is_normal_or_strat & idx_who3_all, 2), score_explore(is_normal_or_strat & idx_who3_all, 3), 15, P.colorWHO3, 's', 'filled', 'MarkerFaceAlpha', 0.3); leg_pca_3d_out{end+1} = 'WHO-3 (Normal)'; end
    if any(flag_OR_outlier), h_pca_3d_out(end+1) = scatter3(ax_pca_3d_outlier, score_explore(flag_OR_outlier, 1), score_explore(flag_OR_outlier, 2), score_explore(flag_OR_outlier, 3), 25, P.colorOutlierGeneric, 'x'); leg_pca_3d_out{end+1} = 'Outlier (T2 or Q)'; end
    hold(ax_pca_3d_outlier, 'off'); view(ax_pca_3d_outlier, -30, 20);
    xlabel(ax_pca_3d_outlier, sprintf('PC1 (%.1f%%)', explained_explore(1))); 
    ylabel(ax_pca_3d_outlier, sprintf('PC2 (%.1f%%)', explained_explore(2)));
    zlabel(ax_pca_3d_outlier, sprintf('PC3 (%.1f%%)', explained_explore(3)));
    title(ax_pca_3d_outlier, 'PC1-PC2-PC3 (Outliers Marked)', 'FontWeight','normal');
    if ~isempty(h_pca_3d_out), legend(ax_pca_3d_outlier, h_pca_3d_out, leg_pca_3d_out, 'Location', 'best', 'FontSize', P.plotFontSize-2); end
    grid(ax_pca_3d_outlier,'on'); axis(ax_pca_3d_outlier,'tight'); set(ax_pca_3d_outlier, 'FontSize', P.plotFontSize-1);
else
    text(0.5,0.5, 'Less than 3 PCs available', 'Parent', ax_pca_3d_outlier, 'HorizontalAlignment','center');
    title(ax_pca_3d_outlier, 'PC1-PC2-PC3 (Outliers Marked)', 'FontWeight','normal');
end
exportgraphics(fig4_pca_scores_combined, fullfile(figuresDir, sprintf('%s_Vis4_PCA_Score_Plots_Combined.tiff', P.datePrefix)), 'Resolution', 300);
savefig(fig4_pca_scores_combined, fullfile(figuresDir, sprintf('%s_Vis4_PCA_Score_Plots_Combined.fig', P.datePrefix)));
fprintf('Plot 4 (Combined PCA Score Plots) saved.\n');


% Plot 5: All PC Loadings (PC1 to k_model_explore) in a single tiled layout
num_pcs_for_loadings = k_model_explore; 
if num_pcs_for_loadings > 0 && ~isempty(coeff_explore)
    if num_pcs_for_loadings <= 4, ncols_loadings = 1; nrows_loadings = num_pcs_for_loadings;
    elseif num_pcs_for_loadings <= 8, ncols_loadings = 2; nrows_loadings = ceil(num_pcs_for_loadings / ncols_loadings);
    else, ncols_loadings = 3; nrows_loadings = ceil(num_pcs_for_loadings / ncols_loadings); end
    
    fig5_loadings_combined = figure('Name', sprintf('Exploratory: PCA Loadings (PC1 to PC%d of T2/Q Model)', num_pcs_for_loadings));
    fig5_loadings_combined.Position = [100, 50, min(ncols_loadings * 450, 1350), min(nrows_loadings * 220, 880)];
    tl_loadings = tiledlayout(nrows_loadings, ncols_loadings, 'TileSpacing','compact', 'Padding','tight');
    title(tl_loadings, sprintf('PCA Loadings for %d PCs used in T2/Q Model', num_pcs_for_loadings), 'FontSize', P.plotFontSize+1);
    for pc_idx = 1:num_pcs_for_loadings
        if pc_idx > size(coeff_explore,2), break; end 
        ax_l = nexttile(tl_loadings);
        plot(ax_l, wavenumbers_roi, coeff_explore(:, pc_idx), 'LineWidth',1); 
        title(ax_l, sprintf('PC%d Loadings (Expl.Var: %.2f%%)', pc_idx, explained_explore(pc_idx)), 'FontWeight','normal', 'FontSize', P.plotFontSize-1); 
        ylabel(ax_l, 'Loading Value', 'FontSize', P.plotFontSize-2); grid(ax_l,'on'); 
        set(ax_l, 'XDir','reverse', 'XLim', P.plotXLim, 'FontSize', P.plotFontSize-2);
        current_tile_info = get(ax_l,'Layout'); % Get current tile info
        current_col = mod(current_tile_info.Tile-1, ncols_loadings) + 1;
        current_row = ceil(current_tile_info.Tile / ncols_loadings);
        if current_col ~= 1, set(ax_l, 'YTickLabel',[]); end
        if current_row ~= nrows_loadings, set(ax_l, 'XTickLabel',[]); end
    end
    xlabel(tl_loadings, P.plotXLabel, 'FontSize', P.plotFontSize-1);
    exportgraphics(fig5_loadings_combined, fullfile(figuresDir, sprintf('%s_Vis5_PCA_Loadings_kModel.tiff', P.datePrefix)), 'Resolution', 300);
    savefig(fig5_loadings_combined, fullfile(figuresDir, sprintf('%s_Vis5_PCA_Loadings_kModel.fig', P.datePrefix)));
    fprintf('Plot 5 (Combined PCA Loadings for k_model PCs) saved.\n');
else
    fprintf('Skipping Plot 5 (Combined PCA Loadings) as k_model_explore is 0 or coeffs are empty.\n');
end

fprintf('All requested diagnostic visualizations generated and saved.\n');
fprintf('Please review figures in: %s\n', figuresDir);

%% --- 4. Save Exploratory Analysis Data ---
% (This section remains the same as in the previous full script version, 
%  saving T2_values, Q_values, all flags, PCA model, etc. using _explore suffixed variables)
fprintf('\n--- 4. Saving Exploratory Analysis Data (T2/Q values, flags, PCA model) ---\n');
exploratoryOutlierData = struct();
exploratoryOutlierData.scriptRunDate = P.datePrefix;
exploratoryOutlierData.alpha_significance = P.alpha_significance;
exploratoryOutlierData.variance_to_explain_for_PCA_model = P.variance_to_explain_for_PCA_model;
exploratoryOutlierData.k_model = k_model_explore; % Note: Renamed to k_model for consistency if loaded elsewhere

exploratoryOutlierData.T2_values = T2_values; % Renamed from T2_values_explore for easier use
exploratoryOutlierData.T2_threshold = T2_threshold; % Renamed
exploratoryOutlierData.flag_T2_outlier = flag_T2_outlier; % Renamed

exploratoryOutlierData.Q_values = Q_values; % Renamed
exploratoryOutlierData.Q_threshold = Q_threshold; % Renamed
exploratoryOutlierData.flag_Q_outlier = flag_Q_outlier; % Renamed

exploratoryOutlierData.flag_AND_outlier = is_T2_and_Q; % Using the specific category flag
exploratoryOutlierData.flag_OR_outlier = flag_T2_outlier | flag_Q_outlier;  

exploratoryOutlierData.Original_ProbeRowIndices = Original_ProbeRowIndices_explore;
exploratoryOutlierData.Original_SpectrumIndexInProbe = Original_SpectrumIndexInProbe_explore;
exploratoryOutlierData.Patient_ID = Patient_ID_explore; % Renamed
exploratoryOutlierData.y_numeric = y_numeric_explore; % Renamed
exploratoryOutlierData.y_categorical = y_cat_explore; % Renamed

exploratoryOutlierData.PCA_coeff = coeff_explore;
exploratoryOutlierData.PCA_mu = mu_explore;
exploratoryOutlierData.PCA_latent = latent_explore;
exploratoryOutlierData.PCA_explained = explained_explore;
exploratoryOutlierData.PCA_scores = score_explore; % Renamed for clarity

exploratoryFilename_mat = fullfile(resultsDir, sprintf('%s_ExploratoryOutlier_AnalysisData.mat', P.datePrefix));
save(exploratoryFilename_mat, 'exploratoryOutlierData', '-v7.3');
fprintf('Exploratory outlier analysis data saved to: %s\n', exploratoryFilename_mat);

fprintf('\n--- Exploratory Outlier Visualization Script Finished ---\n');
fprintf('Review generated plots and data. Next step: Decide on removal strategy and apply in a separate script or function.\n');