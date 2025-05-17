% run_exploratory_outlier_visualization.m
%
% PURPOSE:
%   Generates a specific set of exploratory visualizations for outlier
%   analysis using PCA, Hotelling's T-squared, and Q-residuals.
%   This script focuses on visualization to inform subsequent outlier
%   removal decisions and does NOT perform any data removal itself.
%
% INPUTS (expected in workspace or loaded from .mat):
%   - dataTableTrain: Table containing probe-level training data, with
%                     'CombinedSpectra', 'WHO_Grade', and 'Diss_ID'.
%                     'CombinedSpectra' should contain preprocessed spectra.
%   - wavenumbers_roi: Vector of wavenumber values.
%
% OUTPUTS (saved to files):
%   - Specific diagnostic plots (.tiff and .fig).
%   - A .mat file with T2/Q values, thresholds, flags, PCA model info.
%
% DATE: 2025-05-17

%% --- 0. Configuration & Setup ---
clearvars -except dataTableTrain wavenumbers_roi; % Keep inputs if already in workspace
close all; % Close any existing figures
fprintf('Starting Focused Exploratory Outlier Visualization - %s\n', string(datetime('now')));

% --- Define Paths (User to verify/modify) ---
projectBasePath = 'C:\Users\Franz\OneDrive\01_Promotion\01 Data\meningioma-ftir-classification'; 
dataDir    = fullfile(projectBasePath, 'data');
resultsDir = fullfile(projectBasePath, 'results', 'Phase1_ExploratoryOutlierVis_Focused'); % New specific subfolder
figuresDir = fullfile(projectBasePath, 'figures', 'Phase1_ExploratoryOutlierVis_Focused'); % New specific subfolder

if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
if ~exist(figuresDir, 'dir'), mkdir(figuresDir); end

% --- Parameters ---
P.alpha_significance = 0.01; 
P.variance_to_explain_for_PCA_model = 0.95; 
P.datePrefix = string(datetime('now','Format','yyyyMMdd'));

% --- Plotting Defaults ---
P.colorWHO1 = [0.9, 0.6, 0.4]; 
P.colorWHO3 = [0.4, 0.702, 0.902]; 
P.colorT2OutlierFlag = [0.8, 0.2, 0.2];   
P.colorQOutlierFlag = [0.2, 0.2, 0.8];    
P.colorBothOutlierFlag = [0.8, 0, 0.8]; 
P.colorGenericOutlierMarker = [0.6 0.6 0.6]; % For the "outliers marked" PCA plots

P.plotFontSize = 10;
P.plotXLabel = 'Wellenzahl (cm^{-1})';
P.plotYLabelAbsorption = 'Absorption (a.u.)';
P.plotXLim = [950 1800];

fprintf('Setup complete. Results will be saved in:\n  %s\n  %s\n', resultsDir, figuresDir);

%% --- 1. Load and Prepare Training Data ---
fprintf('\n--- 1. Loading and Preparing Training Data ---\n');
if ~exist('dataTableTrain', 'var')
    trainDataTableFile = fullfile(dataDir, 'data_table_train.mat'); 
    if exist(trainDataTableFile, 'file')
        fprintf('Loading dataTableTrain from: %s\n', trainDataTableFile);
        loadedVars = load(trainDataTableFile);
        if isfield(loadedVars, 'dataTableTrain'), dataTableTrain = loadedVars.dataTableTrain;
        else, error('Variable "dataTableTrain" not found within %s.', trainDataTableFile); end
    else, error('Input file %s not found AND dataTableTrain not in workspace.', trainDataTableFile); end
end
fprintf('dataTableTrain available with %d probes.\n', height(dataTableTrain));
if ~exist('wavenumbers_roi', 'var')
    try load(fullfile(dataDir, 'wavenumbers.mat'), 'wavenumbers_roi');
        if iscolumn(wavenumbers_roi), wavenumbers_roi = wavenumbers_roi'; end
    catch ME_wave, error('Error loading wavenumbers.mat: %s', ME_wave.message); end
end
fprintf('wavenumbers_roi available (%d points).\n', length(wavenumbers_roi));

allSpectra_cell = {}; allLabels_cell = {};
allOriginalProbeRowIndices_vec = []; allOriginalSpectrumIndices_InProbe_vec = []; 
fprintf('Extracting and flattening spectra from dataTableTrain.CombinedSpectra...\n');
for i = 1:height(dataTableTrain)
    spectraMatrix = dataTableTrain.CombinedSpectra{i}; 
    if isempty(spectraMatrix) || ~isnumeric(spectraMatrix) || ndims(spectraMatrix) ~= 2, continue; end
    if size(spectraMatrix,1) == 0, continue; end
    if size(spectraMatrix,2) ~= length(wavenumbers_roi), continue; end
    numIndividualSpectra = size(spectraMatrix, 1);
    allSpectra_cell{end+1,1} = spectraMatrix;
    allLabels_cell{end+1,1} = repmat(dataTableTrain.WHO_Grade(i), numIndividualSpectra, 1);
    allOriginalProbeRowIndices_vec = [allOriginalProbeRowIndices_vec; repmat(i, numIndividualSpectra, 1)];
    allOriginalSpectrumIndices_InProbe_vec = [allOriginalSpectrumIndices_InProbe_vec; (1:numIndividualSpectra)'];
end
if isempty(allSpectra_cell), error('No valid spectra extracted.'); end
X = cell2mat(allSpectra_cell); % Using X for brevity in this script
y_cat = cat(1, allLabels_cell{:});
Original_ProbeRowIndices = allOriginalProbeRowIndices_vec; 
Original_SpectrumIndexInProbe = allOriginalSpectrumIndices_InProbe_vec;
y_numeric = zeros(length(y_cat), 1);
categories_y = categories(y_cat);
idx_who1 = find(strcmp(categories_y, 'WHO-1')); idx_who3 = find(strcmp(categories_y, 'WHO-3'));
if ~isempty(idx_who1), y_numeric(y_cat == categories_y{idx_who1}) = 1; end
if ~isempty(idx_who3), y_numeric(y_cat == categories_y{idx_who3}) = 3; end
fprintf('Data for analysis: %d spectra selected, %d features.\n', size(X,1), size(X,2)); % User request

%% --- 2. Perform PCA, Calculate T² and Q Statistics & Thresholds ---
fprintf('\n--- 2. PCA, T2/Q Calculation & Thresholds ---\n');
[coeff, score, latent, ~, explained, mu] = pca(X, 'Algorithm','svd');
fprintf('PCA completed. Variance by 1st PC: %.2f%%\n', explained(1));
cumulativeVariance = cumsum(explained);
k_model = find(cumulativeVariance >= P.variance_to_explain_for_PCA_model*100, 1, 'first');
if isempty(k_model), k_model = min(length(explained), size(X,1)-1); end
if k_model == 0 && ~isempty(explained), k_model = 1; end
if isempty(explained) || k_model == 0, error('Could not determine k_model for T2/Q.'); end
fprintf('k_model (for T2/Q) based on >=%.0f%% variance: %d PCs.\n', P.variance_to_explain_for_PCA_model*100, k_model);

score_k = score(:, 1:k_model); lambda_k = latent(1:k_model); lambda_k(lambda_k <= eps) = eps;
T2_values = sum(bsxfun(@rdivide, score_k.^2, lambda_k'), 2);
n_samples = size(X, 1);
if n_samples > k_model && k_model > 0
    T2_threshold = ((k_model * (n_samples - 1)) / (n_samples - k_model)) * finv(1 - P.alpha_significance, k_model, n_samples - k_model);
else, T2_threshold = chi2inv(1 - P.alpha_significance, k_model); end
fprintf('Hotelling T2 threshold (alpha=%.2f): %.4f\n', P.alpha_significance, T2_threshold);

X_reconstructed = score(:, 1:k_model) * coeff(:, 1:k_model)' + mu;
Q_values = sum((X - X_reconstructed).^2, 2);
num_total_pcs = min(size(X)-[1 0]); % Max possible PCs is min(N-1, P)
Q_threshold = NaN;
if k_model < num_total_pcs && k_model < length(latent) % Ensure there are discarded eigenvalues
    discarded_eigenvalues = latent(k_model+1:min(length(latent), num_total_pcs)); % Ensure index is valid
    discarded_eigenvalues(discarded_eigenvalues <= eps) = eps;
    theta1 = sum(discarded_eigenvalues); theta2 = sum(discarded_eigenvalues.^2); theta3 = sum(discarded_eigenvalues.^3);
    if theta1 > eps && theta2 > eps
        h0 = 1 - (2 * theta1 * theta3) / (3 * theta2^2);
        if h0 <= eps, h0 = 1; end
        ca = norminv(1 - P.alpha_significance);
        val_in_bracket = ca * sqrt(2*theta2*h0^2)/theta1 + 1 + theta2*h0*(h0-1)/(theta1^2);
        if val_in_bracket > 0, Q_threshold = theta1 * (val_in_bracket)^(1/h0); end
    end
end
if isnan(Q_threshold) || isinf(Q_threshold)
    Q_threshold = prctile(Q_values, (1-P.alpha_significance)*100);
    fprintf('Using empirical Q-threshold: %.4g\n', Q_threshold);
else, fprintf('Q-Statistic (SPE) threshold (alpha=%.2f): %.4g\n', P.alpha_significance, Q_threshold); end

flag_T2 = (T2_values > T2_threshold); flag_Q = (Q_values > Q_threshold);
is_T2_only = flag_T2 & ~flag_Q; is_Q_only = ~flag_T2 & flag_Q;
is_T2_and_Q = flag_T2 & flag_Q; is_normal = ~flag_T2 & ~flag_Q;

%% --- 3. Generate Requested Visualizations ---
fprintf('\n--- 3. Generating Requested Visualizations ---\n');

% PLOT 1: Q-Statistic / T2 Statistic individual Plots in a Tiled Layout
fig1 = figure('Name', 'Individual T2 and Q Statistics'); fig1.Position = [50, 500, 900, 600];
tl1 = tiledlayout(2,1,'TileSpacing','compact','Padding','compact'); title(tl1,'T2 & Q Statistics');
ax1a = nexttile(tl1); plot(ax1a, T2_values, 'o', 'MarkerSize',3); hold on; plot(ax1a, find(flag_T2),T2_values(flag_T2),'x','Color',P.colorT2OutlierFlag); yline(ax1a, T2_threshold,'r--'); ylabel(ax1a,'T2'); title(ax1a, 'Hotelling T^2'); grid on;
ax1b = nexttile(tl1); plot(ax1b, Q_values, 'o', 'MarkerSize',3); hold on; plot(ax1b, find(flag_Q),Q_values(flag_Q),'x','Color',P.colorQOutlierFlag); yline(ax1b, Q_threshold,'m--'); ylabel(ax1b,'Q'); title(ax1b, 'Q-Statistic (SPE)'); grid on; xlabel(ax1b,'Spectrum Index');
exportgraphics(fig1, fullfile(figuresDir, sprintf('%s_Vis1_T2_Q_Individual.tiff',P.datePrefix)),'Resolution',300); savefig(fig1, fullfile(figuresDir,sprintf('%s_Vis1_T2_Q_Individual.fig',P.datePrefix))); fprintf('Plot 1 (Individual T2 & Q) saved.\n');

% PLOT 2: T2 vs Q Plot with Thresholds and Outliers (Categorized)
fig2 = figure('Name', 'T2 vs Q with Categories & Thresholds'); fig2.Position = [100 100 750 600]; hold on;
hdl2 = []; leg2 = {};
if any(is_normal & y_numeric==1),hdl2(end+1)=plot(T2_values(is_normal & y_numeric==1), Q_values(is_normal & y_numeric==1), 'o','MarkerSize',4,'MarkerFaceColor',P.colorWHO1,'Color',P.colorWHO1); leg2{end+1}='WHO1(Normal)'; end
if any(is_normal & y_numeric==3),hdl2(end+1)=plot(T2_values(is_normal & y_numeric==3), Q_values(is_normal & y_numeric==3), 's','MarkerSize',4,'MarkerFaceColor',P.colorWHO3,'Color',P.colorWHO3); leg2{end+1}='WHO3(Normal)'; end
if any(is_T2_only), hdl2(end+1)=plot(T2_values(is_T2_only), Q_values(is_T2_only), 's','MarkerSize',5,'Color',P.colorT2OutlierFlag); leg2{end+1}='T2-only'; end
if any(is_Q_only), hdl2(end+1)=plot(T2_values(is_Q_only), Q_values(is_Q_only), 'd','MarkerSize',5,'Color',P.colorQOutlierFlag); leg2{end+1}='Q-only'; end
if any(is_T2_and_Q), hdl2(end+1)=plot(T2_values(is_T2_and_Q), Q_values(is_T2_and_Q), '*','MarkerSize',6,'Color',P.colorBothOutlierFlag); leg2{end+1}='T2&Q'; end
line([T2_threshold T2_threshold],get(gca,'YLim'),'Color',P.colorT2OutlierFlag,'LineStyle','--','LineWidth',1,'HandleVisibility','off');
line(get(gca,'XLim'),[Q_threshold Q_threshold],'Color',P.colorQOutlierFlag,'LineStyle','--','LineWidth',1,'HandleVisibility','off');
hold off; xlabel('Hotelling T^2'); ylabel('Q-Statistic (SPE)'); title('T^2 vs. Q with Outlier Categories & Thresholds');
if ~isempty(hdl2), legend(hdl2,leg2,'Location','NorthEast','FontSize',P.plotFontSize-1); end; grid on;
exportgraphics(fig2, fullfile(figuresDir, sprintf('%s_Vis2_T2vQ_CategoriesAndThresholds.tiff',P.datePrefix)),'Resolution',300); savefig(fig2, fullfile(figuresDir,sprintf('%s_Vis2_T2vQ_CategoriesAndThresholds.fig',P.datePrefix))); fprintf('Plot 2 (T2vQ Categories & Thresh) saved.\n');

% PLOT 3: T2 vs Q Plot only WHO-1 and WHO-3 (Raw Distribution, No Outlier Marks, No Thresholds)
fig3 = figure('Name', 'T2 vs Q Raw Distribution by WHO Grade'); fig3.Position = [150 150 750 600]; hold on;
hdl3 = []; leg3 = {};
idx_w1 = (y_numeric == 1); idx_w3 = (y_numeric == 3); idx_unk = ~(idx_w1 | idx_w3);
if any(idx_w1),hdl3(end+1)=scatter(T2_values(idx_w1), Q_values(idx_w1), 15, P.colorWHO1, 'o', 'filled', 'MarkerFaceAlpha',0.4); leg3{end+1}='WHO-1'; end
if any(idx_w3),hdl3(end+1)=scatter(T2_values(idx_w3), Q_values(idx_w3), 15, P.colorWHO3, 's', 'filled', 'MarkerFaceAlpha',0.4); leg3{end+1}='WHO-3'; end
if any(idx_unk),hdl3(end+1)=scatter(T2_values(idx_unk), Q_values(idx_unk), 15, [0.7 0.7 0.7], '^', 'filled', 'MarkerFaceAlpha',0.3); leg3{end+1}='Other'; end
hold off; xlabel('Hotelling T^2 Value'); ylabel('Q-Statistic (SPE) Value'); title('T^2 vs. Q Raw Distribution by WHO Grade');
if ~isempty(hdl3), legend(hdl3,leg3,'Location','best','FontSize',P.plotFontSize-1); end; grid on;
exportgraphics(fig3, fullfile(figuresDir, sprintf('%s_Vis3_T2vQ_WHOgradesOnly.tiff',P.datePrefix)),'Resolution',300); savefig(fig3, fullfile(figuresDir,sprintf('%s_Vis3_T2vQ_WHOgradesOnly.fig',P.datePrefix))); fprintf('Plot 3 (T2vQ Raw WHO Grades) saved.\n');

% PLOT 4a: PCA Scores (WHO Grade Only) - PC1vsPC2 and 3D PC1-3
fig4a = figure('Name', 'PCA Scores by WHO Grade'); fig4a.Position = [50 50 900 450];
tl4a = tiledlayout(1,2,'TileSpacing','compact','Padding','compact'); title(tl4a,'PCA Scores by WHO Grade');
% PC1 vs PC2 - WHO Grade Only
ax4a1 = nexttile(tl4a); hold(ax4a1, 'on'); hdl4a1=[]; leg4a1={};
if any(idx_w1),hdl4a1(end+1)=scatter(ax4a1,score(idx_w1,1),score(idx_w1,min(2,size(score,2))),15,P.colorWHO1,'o','filled','MarkerFaceAlpha',0.3); leg4a1{end+1}='WHO-1'; end
if any(idx_w3),hdl4a1(end+1)=scatter(ax4a1,score(idx_w3,1),score(idx_w3,min(2,size(score,2))),15,P.colorWHO3,'s','filled','MarkerFaceAlpha',0.3); leg4a1{end+1}='WHO-3'; end
if any(idx_unk),hdl4a1(end+1)=scatter(ax4a1,score(idx_unk,1),score(idx_unk,min(2,size(score,2))),15,[0.7 0.7 0.7],'^','filled','MarkerFaceAlpha',0.2); leg4a1{end+1}='Other'; end
hold(ax4a1,'off'); xlabel(ax4a1,sprintf('PC1 (%.1f%%)',explained(1))); ylabel(ax4a1,sprintf('PC2 (%.1f%%)',explained(min(2,length(explained))))); title(ax4a1,'PC1 vs PC2'); 
if ~isempty(hdl4a1), legend(ax4a1,hdl4a1,leg4a1,'Location','best','FontSize',P.plotFontSize-2); end; axis(ax4a1,'equal');grid(ax4a1,'on');
% PC1 vs PC2 vs PC3 - WHO Grade Only
ax4a2 = nexttile(tl4a);
if size(score,2)>=3
    hold(ax4a2, 'on'); hdl4a2=[]; leg4a2={};
    if any(idx_w1),hdl4a2(end+1)=scatter3(ax4a2,score(idx_w1,1),score(idx_w1,2),score(idx_w1,3),15,P.colorWHO1,'o','filled','MarkerFaceAlpha',0.3); leg4a2{end+1}='WHO-1'; end
    if any(idx_w3),hdl4a2(end+1)=scatter3(ax4a2,score(idx_w3,1),score(idx_w3,2),score(idx_w3,3),15,P.colorWHO3,'s','filled','MarkerFaceAlpha',0.3); leg4a2{end+1}='WHO-3'; end
    if any(idx_unk),hdl4a2(end+1)=scatter3(ax4a2,score(idx_unk,1),score(idx_unk,2),score(idx_unk,3),15,[0.7 0.7 0.7],'^','filled','MarkerFaceAlpha',0.2); leg4a2{end+1}='Other'; end
    hold(ax4a2,'off'); view(ax4a2, -30, 20); xlabel(ax4a2,sprintf('PC1 (%.1f%%)',explained(1))); ylabel(ax4a2,sprintf('PC2 (%.1f%%)',explained(2))); zlabel(ax4a2,sprintf('PC3 (%.1f%%)',explained(3))); title(ax4a2,'PC1-PC2-PC3 (3D)');
    if ~isempty(hdl4a2), legend(ax4a2,hdl4a2,leg4a2,'Location','best','FontSize',P.plotFontSize-2); end; grid(ax4a2,'on'); axis(ax4a2,'tight');
else, text(0.5,0.5,'<3 PCs','Parent',ax4a2,'HorizontalAlignment','center'); title(ax4a2,'PC1-PC2-PC3 (3D)'); end
exportgraphics(fig4a, fullfile(figuresDir, sprintf('%s_Vis4a_PCA_Scores_WHOonly.tiff',P.datePrefix)),'Resolution',300); savefig(fig4a, fullfile(figuresDir,sprintf('%s_Vis4a_PCA_Scores_WHOonly.fig',P.datePrefix))); fprintf('Plot 4a (PCA Scores WHO only) saved.\n');

% PLOT 4b: PCA Scores (Outliers Marked by Category) - PC1vsPC2 and 3D PC1-3
fig4b = figure('Name', 'PCA Scores with Outlier Categories'); fig4b.Position = [100 50 900 450];
tl4b = tiledlayout(1,2,'TileSpacing','compact','Padding','compact'); title(tl4b,'PCA Scores with Outlier Categories');
% PC1 vs PC2 - Outliers Marked
ax4b1 = nexttile(tl4b); hold(ax4b1,'on'); hdl4b1=[]; leg4b1={};
if any(is_normal & y_numeric==1),hdl4b1(end+1)=scatter(ax4b1,score(is_normal & y_numeric==1,1),score(is_normal & y_numeric==1,min(2,size(score,2))),15,P.colorWHO1,'o','filled','MarkerFaceAlpha',0.3); leg4b1{end+1}='WHO1(Normal)'; end
if any(is_normal & y_numeric==3),hdl4b1(end+1)=scatter(ax4b1,score(is_normal & y_numeric==3,1),score(is_normal & y_numeric==3,min(2,size(score,2))),15,P.colorWHO3,'s','filled','MarkerFaceAlpha',0.3); leg4b1{end+1}='WHO3(Normal)'; end
if any(is_T2_only),hdl4b1(end+1)=scatter(ax4b1,score(is_T2_only,1),score(is_T2_only,min(2,size(score,2))),25,P.colorT2OutlierFlag,'s'); leg4b1{end+1}='T2-only';end
if any(is_Q_only),hdl4b1(end+1)=scatter(ax4b1,score(is_Q_only,1),score(is_Q_only,min(2,size(score,2))),25,P.colorQOutlierFlag,'d'); leg4b1{end+1}='Q-only';end
if any(is_T2_and_Q),hdl4b1(end+1)=scatter(ax4b1,score(is_T2_and_Q,1),score(is_T2_and_Q,min(2,size(score,2))),30,P.colorBothOutlierFlag,'*'); leg4b1{end+1}='T2&Q';end
hold(ax4b1,'off'); xlabel(ax4b1,sprintf('PC1 (%.1f%%)',explained(1))); ylabel(ax4b1,sprintf('PC2 (%.1f%%)',explained(min(2,length(explained))))); title(ax4b1,'PC1 vs PC2 (Outliers by Category)');
if ~isempty(hdl4b1),legend(ax4b1,hdl4b1,leg4b1,'Location','best','FontSize',P.plotFontSize-2);end; axis(ax4b1,'equal');grid(ax4b1,'on');
% PC1 vs PC2 vs PC3 - Outliers Marked
ax4b2 = nexttile(tl4b);
if size(score,2)>=3
    hold(ax4b2,'on'); hdl4b2=[]; leg4b2={};
    if any(is_normal & y_numeric==1),hdl4b2(end+1)=scatter3(ax4b2,score(is_normal & y_numeric==1,1),score(is_normal & y_numeric==1,2),score(is_normal & y_numeric==1,3),15,P.colorWHO1,'o','filled','MarkerFaceAlpha',0.3); leg4b2{end+1}='WHO1(Normal)';end
    if any(is_normal & y_numeric==3),hdl4b2(end+1)=scatter3(ax4b2,score(is_normal & y_numeric==3,1),score(is_normal & y_numeric==3,2),score(is_normal & y_numeric==3,3),15,P.colorWHO3,'s','filled','MarkerFaceAlpha',0.3); leg4b2{end+1}='WHO3(Normal)';end
    if any(is_T2_only),hdl4b2(end+1)=scatter3(ax4b2,score(is_T2_only,1),score(is_T2_only,2),score(is_T2_only,3),25,P.colorT2OutlierFlag,'s'); leg4b2{end+1}='T2-only';end
    if any(is_Q_only),hdl4b2(end+1)=scatter3(ax4b2,score(is_Q_only,1),score(is_Q_only,2),score(is_Q_only,3),25,P.colorQOutlierFlag,'d'); leg4b2{end+1}='Q-only';end
    if any(is_T2_and_Q),hdl4b2(end+1)=scatter3(ax4b2,score(is_T2_and_Q,1),score(is_T2_and_Q,2),score(is_T2_and_Q,3),30,P.colorBothOutlierFlag,'*'); leg4b2{end+1}='T2&Q';end
    hold(ax4b2,'off'); view(ax4b2, -30, 20); xlabel(ax4b2,sprintf('PC1 (%.1f%%)',explained(1))); ylabel(ax4b2,sprintf('PC2 (%.1f%%)',explained(2))); zlabel(ax4b2,sprintf('PC3 (%.1f%%)',explained(3))); title(ax4b2,'PC1-PC2-PC3 (Outliers by Category)');
    if ~isempty(hdl4b2),legend(ax4b2,hdl4b2,leg4b2,'Location','best','FontSize',P.plotFontSize-2);end; grid(ax4b2,'on'); axis(ax4b2,'tight');
else, text(0.5,0.5,'<3 PCs','Parent',ax4b2,'HorizontalAlignment','center'); title(ax4b2,'PC1-PC2-PC3 (Outliers by Category)'); end
exportgraphics(fig4b, fullfile(figuresDir, sprintf('%s_Vis4b_PCA_Scores_OutlierCats.tiff',P.datePrefix)),'Resolution',300); savefig(fig4b, fullfile(figuresDir,sprintf('%s_Vis4b_PCA_Scores_OutlierCats.fig',P.datePrefix))); fprintf('Plot 4b (PCA Scores Outlier Cats) saved.\n');

% PLOT 5: All PC Loadings (PC1 to k_model) in a single tiled layout
% (Code for this from previous response, using k_model, coeff, explained, P, wavenumbers_roi)
% (Ensure it uses the correct variable names as defined in this script's Section 2)
num_pcs_for_loadings = k_model; 
if num_pcs_for_loadings > 0 && ~isempty(coeff)
    if num_pcs_for_loadings <= 3, ncols_loadings = 1; nrows_loadings = num_pcs_for_loadings;
    elseif num_pcs_for_loadings <= 8, ncols_loadings = 2; nrows_loadings = ceil(num_pcs_for_loadings / ncols_loadings);
    else, ncols_loadings = 3; nrows_loadings = ceil(num_pcs_for_loadings / ncols_loadings); end
    fig5 = figure('Name',sprintf('PCA Loadings (PC1-PC%d of T2/Q Model)',num_pcs_for_loadings));fig5.Position = [100 50 min(ncols_loadings*450,1350) min(nrows_loadings*220,880)];
    tl5 = tiledlayout(nrows_loadings,ncols_loadings,'TileSpacing','compact','Padding','tight'); title(tl5,sprintf('PCA Loadings for %d PCs used in T2/Q Model',num_pcs_for_loadings));
    for pc_idx = 1:num_pcs_for_loadings
        if pc_idx > size(coeff,2), break; end; ax_l = nexttile(tl5); plot(ax_l,wavenumbers_roi,coeff(:,pc_idx),'LineWidth',1); title(ax_l,sprintf('PC%d Loadings (Expl.Var: %.2f%%)',pc_idx,explained(pc_idx)),'FontWeight','normal','FontSize',P.plotFontSize-1); ylabel(ax_l,'Loading Value','FontSize',P.plotFontSize-2);grid(ax_l,'on');set(ax_l,'XDir','reverse','XLim',P.plotXLim,'FontSize',P.plotFontSize-2);
        current_tile_info=get(ax_l,'Layout');current_col=mod(current_tile_info.Tile-1,ncols_loadings)+1;current_row=ceil(current_tile_info.Tile/ncols_loadings);
        if current_col~=1,set(ax_l,'YTickLabel',[]);end;if current_row~=nrows_loadings,set(ax_l,'XTickLabel',[]);end
    end; xlabel(tl5,P.plotXLabel,'FontSize',P.plotFontSize-1);
    exportgraphics(fig5,fullfile(figuresDir,sprintf('%s_Vis5_PCA_Loadings_kModel.tiff',P.datePrefix)),'Resolution',300);savefig(fig5,fullfile(figuresDir,sprintf('%s_Vis5_PCA_Loadings_kModel.fig',P.datePrefix)));fprintf('Plot 5 (PCA Loadings kModel) saved.\n');
else, fprintf('Skipping Plot 5 (Loadings) as k_model is 0 or coeffs empty.\n'); end

% PLOT 6: Tiled Layout of Spectra for Distinct Outlier Categories (Q-only, T2-only, T2&Q)
% (Code for this from previous response - Plot 3.8 logic)
% (Ensure it uses is_Q_only, is_T2_only, is_T2_and_Q, X, wavenumbers_roi, P)
fig6 = figure('Name', 'Spectra of Distinct Outlier Categories'); fig6.Position = [120 120 700 850];
tl6 = tiledlayout(3,1,'TileSpacing','compact','Padding','compact'); title(tl6,'Spectra by Distinct Outlier Category');
outlier_cats_plot6 = {{'Q-only Flagged',is_Q_only,P.colorQOutlierFlag},{'T2-only Flagged',is_T2_only,P.colorT2OutlierFlag},{'T2&Q Flagged (Consensus)',is_T2_and_Q,P.colorBothOutlierFlag}};
for cat_idx=1:3, ax_cat=nexttile(tl6);hold(ax_cat,'on');cat_title=outlier_cats_plot6{cat_idx}{1};cat_flag=outlier_cats_plot6{cat_idx}{2};cat_color=outlier_cats_plot6{cat_idx}{3};
    spectra_cat=X(cat_flag,:);num_cat=sum(cat_flag);hdl_cat_mean=[];
    if num_cat>0, plot(ax_cat,wavenumbers_roi,spectra_cat','Color',[cat_color,0.15],'LineWidth',0.5,'HandleVisibility','off');mean_spec=mean(spectra_cat,1,'omitnan');if any(~isnan(mean_spec)),hdl_cat_mean=plot(ax_cat,wavenumbers_roi,mean_spec,'Color',cat_color,'LineWidth',2,'DisplayName',sprintf('Mean (n=%d)',num_cat));end
    else, text(0.5,0.5,'No spectra','Parent',ax_cat,'HorizontalAlignment','center');end
    hold(ax_cat,'off');title(ax_cat,sprintf('%s (n=%d)',cat_title,num_cat),'FontWeight','normal');ylabel(ax_cat,P.plotYLabelAbsorption);set(ax_cat,'XDir','reverse','XLim',P.plotXLim,'FontSize',P.plotFontSize-1);grid(ax_cat,'on');
    if cat_idx<3,set(ax_cat,'XTickLabel',[]);else,xlabel(ax_cat,P.plotXLabel);end
    if ~isempty(hdl_cat_mean),legend(ax_cat,hdl_cat_mean,'Location','northeast','FontSize',P.plotFontSize-2);end
end
exportgraphics(fig6,fullfile(figuresDir,sprintf('%s_Vis6_OutlierCategory_Spectra.tiff',P.datePrefix)),'Resolution',300);savefig(fig6,fullfile(figuresDir,sprintf('%s_Vis6_OutlierCategory_Spectra.fig',P.datePrefix)));fprintf('Plot 6 (Outlier Category Spectra) saved.\n');

fprintf('All requested diagnostic visualizations generated.\n');

%% --- 4. Save Exploratory Analysis Data ---
fprintf('\n--- 4. Saving Exploratory Analysis Data ---\n');
exploratoryOutlierData = struct();
exploratoryOutlierData.scriptRunDate = P.datePrefix;
exploratoryOutlierData.alpha_significance = P.alpha_significance;
exploratoryOutlierData.variance_to_explain_for_PCA_model = P.variance_to_explain_for_PCA_model;
exploratoryOutlierData.k_model = k_model; 
exploratoryOutlierData.T2_values = T2_values; 
exploratoryOutlierData.T2_threshold = T2_threshold; 
exploratoryOutlierData.flag_T2_outlier = flag_T2; 
exploratoryOutlierData.Q_values = Q_values;
exploratoryOutlierData.Q_threshold = Q_threshold;
exploratoryOutlierData.flag_Q_outlier = flag_Q;
exploratoryOutlierData.flag_T2_only_outlier = is_T2_only;
exploratoryOutlierData.flag_Q_only_outlier = is_Q_only;
exploratoryOutlierData.flag_AND_outlier = is_T2_and_Q; 
exploratoryOutlierData.flag_OR_outlier = flag_T2 | flag_Q;   
exploratoryOutlierData.Original_ProbeRowIndices = Original_ProbeRowIndices;
exploratoryOutlierData.Original_SpectrumIndexInProbe = Original_SpectrumIndexInProbe;
exploratoryOutlierData.Patient_ID = Patient_ID_explore; % From Section 1
exploratoryOutlierData.y_numeric = y_numeric;
exploratoryOutlierData.y_categorical = y_cat;
exploratoryOutlierData.PCA_coeff = coeff;
exploratoryOutlierData.PCA_mu = mu;
exploratoryOutlierData.PCA_latent = latent;
exploratoryOutlierData.PCA_explained = explained;
exploratoryOutlierData.PCA_scores = score; 
exploratoryFilename_mat = fullfile(resultsDir, sprintf('%s_ExploratoryOutlier_AnalysisData.mat', P.datePrefix));
save(exploratoryFilename_mat, 'exploratoryOutlierData', '-v7.3');
fprintf('Exploratory outlier analysis data saved to: %s\n', exploratoryFilename_mat);

fprintf('\n--- Focused Exploratory Outlier Visualization Script Finished ---\n');