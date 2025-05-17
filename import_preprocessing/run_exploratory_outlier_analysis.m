% run_exploratory_outlier_analysis.m
%
% PURPOSE:
%   Generates a specific set of exploratory visualizations for outlier
%   analysis using PCA, Hotelling's T-squared, and Q-residuals.
%   This script focuses on visualization to inform subsequent outlier
%   removal decisions and does NOT perform any data removal itself.
%   It saves the calculated T2/Q values, thresholds, and PCA model.
%
% INPUTS (expected in workspace or loaded from .mat):
%   - dataTableTrain: Table containing probe-level training data, with
%                     'CombinedSpectra', 'WHO_Grade', and 'Diss_ID'.
%                     'CombinedSpectra' should contain preprocessed spectra.
%   - wavenumbers_roi: Vector of wavenumber values.
%
% OUTPUTS (saved to files):
%   - Specific diagnostic plots (.tiff and .fig).
%   - A .mat file with T2/Q values for ALL spectra, thresholds, flags,
%     PCA model info, and mappings to original data.
%
% DATE: 2025-05-17 (Refocused for exploration and saving analysis data only)

%% --- 0. Configuration & Setup ---
clearvars -except dataTableTrain wavenumbers_roi;
close all;
fprintf('Starting Focused Exploratory Outlier Visualization & Analysis Data Generation - %s\n', string(datetime('now')));

projectBasePath = pwd; % Assumes script is run from project root
dataDir    = fullfile(projectBasePath, 'data');
% Results from this script are primarily figures and one .mat file with analysis data
resultsDir = fullfile(projectBasePath, 'results', 'Phase1_OutlierExploration'); % Specific subfolder for this script's .mat output
figuresDir = fullfile(projectBasePath, 'figures', 'Phase1_OutlierExploration'); % Specific subfolder for this script's figures

if ~isfolder(resultsDir), mkdir(resultsDir); end
if ~isfolder(figuresDir), mkdir(figuresDir); end

P.alpha_T2_Q = 0.0001; % Single alpha for T2 and Q for this exploratory run
P.variance_to_explain_for_PCA_model = 0.95;
P.datePrefix = string(datetime('now','Format','yyyyMMdd')); % For output filenames
P.colorWHO1 = [0.9, 0.6, 0.4]; P.colorWHO3 = [0.4, 0.702, 0.902];
P.colorT2OutlierFlag = [0.8, 0.2, 0.2]; P.colorQOutlierFlag = [0.2, 0.2, 0.8];
P.colorBothOutlierFlag = [0.8, 0, 0.8]; % For T2&Q
P.plotFontSize = 10; P.plotXLabel = 'Wellenzahl (cm^{-1})';
P.plotYLabelAbsorption = 'Absorption (a.u.)'; P.plotXLim = [950 1800];

fprintf('Setup complete. Figures will be saved in: %s\n', figuresDir);
fprintf('Analysis data .mat file will be saved in: %s\n', resultsDir);
fprintf('Using alpha for T2 & Q: %.3f\n', P.alpha_T2_Q);

%% --- 1. Load and Prepare Training Data ---
% (This section remains largely the same as your existing script)
fprintf('\n--- 1. Loading and Preparing Training Data ---\n');
if ~exist('dataTableTrain', 'var')
    trainDataTableFile = fullfile(dataDir, 'data_table_train.mat');
    if exist(trainDataTableFile, 'file')
        fprintf('Loading dataTableTrain from: %s\n', trainDataTableFile);
        loadedVars = load(trainDataTableFile);
        if isfield(loadedVars, 'dataTableTrain'), dataTableTrain = loadedVars.dataTableTrain;
        else, error('Var "dataTableTrain" not found in %s.', trainDataTableFile); end
    else, error('File %s not found AND dataTableTrain not in workspace.', trainDataTableFile); end
end
fprintf('dataTableTrain available with %d probes.\n', height(dataTableTrain));

if ~exist('wavenumbers_roi', 'var')
    try
        wavenumbers_data_loaded = load(fullfile(dataDir, 'wavenumbers.mat'), 'wavenumbers_roi');
        wavenumbers_roi = wavenumbers_data_loaded.wavenumbers_roi;
        if iscolumn(wavenumbers_roi), wavenumbers_roi = wavenumbers_roi'; end
    catch ME_wave
        error('Error loading wavenumbers.mat: %s', ME_wave.message);
    end
end
fprintf('wavenumbers_roi available (%d points).\n', length(wavenumbers_roi));

allSpectra_cell = {}; allLabels_cell = {}; Patient_ID_explore_cell = {};
allOriginalProbeRowIndices_vec = []; allOriginalSpectrumIndices_InProbe_vec = [];
fprintf('Extracting spectra from dataTableTrain.CombinedSpectra...\n'); % Assuming CombinedSpectra contains the preprocessed spectra
for i = 1:height(dataTableTrain)
    % IMPORTANT: Use the spectra that have undergone initial preprocessing
    % (e.g., smoothing, SNV, L2-norm) as per your workflow.
    % If 'CombinedSpectra' in dataTableTrain already holds these, this is fine.
    % If not, you might need to load/generate them here or ensure dataTableTrain is already processed.
    spectraMatrix = dataTableTrain.CombinedSpectra{i};
    
    if isempty(spectraMatrix) || ~isnumeric(spectraMatrix) || ndims(spectraMatrix) ~= 2
        warning('Row %d (Diss_ID: %s): CombinedSpectra is empty or invalid. Skipping.', i, dataTableTrain.Diss_ID{i});
        continue;
    end
    if size(spectraMatrix,1) == 0
        warning('Row %d (Diss_ID: %s): CombinedSpectra contains 0 spectra. Skipping.', i, dataTableTrain.Diss_ID{i});
        continue;
    end
    if size(spectraMatrix,2) ~= length(wavenumbers_roi)
        warning('Row %d (Diss_ID: %s): Wavenumber mismatch (Expected %d, Got %d). Skipping.', i, dataTableTrain.Diss_ID{i}, length(wavenumbers_roi), size(spectraMatrix,2));
        continue;
    end

    numIndSpectra = size(spectraMatrix, 1);
    allSpectra_cell{end+1,1} = spectraMatrix;
    allLabels_cell{end+1,1} = repmat(dataTableTrain.WHO_Grade(i), numIndSpectra, 1);
    Patient_ID_explore_cell{end+1,1} = repmat(dataTableTrain.Diss_ID(i), numIndSpectra, 1);
    allOriginalProbeRowIndices_vec = [allOriginalProbeRowIndices_vec; repmat(i, numIndSpectra, 1)];
    allOriginalSpectrumIndices_InProbe_vec = [allOriginalSpectrumIndices_InProbe_vec; (1:numIndSpectra)'];
end
if isempty(allSpectra_cell), error('No valid spectra extracted for outlier analysis.'); end

X = cell2mat(allSpectra_cell);
y_cat = cat(1, allLabels_cell{:});
Patient_ID = vertcat(Patient_ID_explore_cell{:}); % Cell array of strings
Original_ProbeRowIndices = allOriginalProbeRowIndices_vec;
Original_SpectrumIndexInProbe = allOriginalSpectrumIndices_InProbe_vec;

y_numeric = zeros(length(y_cat), 1);
categories_y = categories(y_cat);
idx_who1 = find(strcmp(categories_y, 'WHO-1')); idx_who3 = find(strcmp(categories_y, 'WHO-3'));
if ~isempty(idx_who1), y_numeric(y_cat == categories_y{idx_who1}) = 1; end
if ~isempty(idx_who3), y_numeric(y_cat == categories_y{idx_who3}) = 3; end
fprintf('Data for analysis: %d spectra selected, %d features.\n', size(X,1), size(X,2));

%% --- 2. Perform PCA, Calculate T² and Q Statistics & Thresholds ---
fprintf('\n--- 2. PCA, T2/Q Calculation & Thresholds ---\n');
[coeff, score, latent, ~, explained, mu] = pca(X, 'Algorithm','svd');
fprintf('PCA completed. Var by 1st PC: %.2f%%\n', explained(1));
cumulativeVariance = cumsum(explained);
k_model = find(cumulativeVariance >= P.variance_to_explain_for_PCA_model*100, 1, 'first');
if isempty(k_model), k_model = min(length(explained), size(X,1)-1); if k_model < 1 && ~isempty(explained), k_model = 1; end; end
if k_model == 0 && ~isempty(explained), k_model = 1; end
if isempty(explained) || k_model == 0, error('Could not determine k_model for T2/Q.'); end
fprintf('k_model (T2/Q) for >=%.0f%% var: %d PCs.\n', P.variance_to_explain_for_PCA_model*100, k_model);

score_k = score(:, 1:k_model); lambda_k = latent(1:k_model); lambda_k(lambda_k <= eps) = eps;
T2_values = sum(bsxfun(@rdivide, score_k.^2, lambda_k'), 2);
n_samples = size(X, 1);

if n_samples > k_model && k_model > 0
    T2_threshold = ((k_model*(n_samples-1))/(n_samples-k_model))*finv(1-P.alpha_T2_Q,k_model,n_samples-k_model);
else
    T2_threshold = chi2inv(1-P.alpha_T2_Q,k_model);
end
fprintf('T2 threshold (alpha=%.3f): %.4f\n', P.alpha_T2_Q, T2_threshold);

X_reconstructed = score(:,1:k_model)*coeff(:,1:k_model)' + mu;
Q_values = sum((X - X_reconstructed).^2, 2);
num_total_pcs = min(size(X,1)-1, size(X,2));
Q_threshold = NaN;
if k_model < num_total_pcs && k_model < length(latent) % Ensure k_model is less than available latent values
    discarded_eigenvalues = latent(k_model+1:min(length(latent),num_total_pcs));
    discarded_eigenvalues(discarded_eigenvalues <= eps) = eps;
    theta1=sum(discarded_eigenvalues); theta2=sum(discarded_eigenvalues.^2); theta3=sum(discarded_eigenvalues.^3);
    if theta1>eps && theta2>eps
        h0=1-(2*theta1*theta3)/(3*theta2^2);
        if h0<=eps || isnan(h0) || isinf(h0), h0=1; fprintf('Warning: h0 for Q-thresh invalid. Using h0=1.\n'); end
        ca=norminv(1-P.alpha_T2_Q);
        val_in_bracket=ca*sqrt(2*theta2*h0^2)/theta1+1+theta2*h0*(h0-1)/(theta1^2);
        if val_in_bracket>0,Q_threshold=theta1*(val_in_bracket)^(1/h0);
        else, fprintf('Warning: Val in bracket for Q-thresh non-positive. Using empirical.\n'); Q_threshold = NaN; end
    end
end
if isnan(Q_threshold)||isinf(Q_threshold)
    Q_threshold=prctile(Q_values,(1-P.alpha_T2_Q)*100);
    fprintf('Empirical Q-thresh (alpha=%.3f for percentile): %.4g\n', P.alpha_T2_Q,Q_threshold);
else
    fprintf('Q-SPE thresh (alpha=%.3f): %.4g\n',P.alpha_T2_Q,Q_threshold);
end

flag_T2 = (T2_values > T2_threshold);
flag_Q = (Q_values > Q_threshold);
is_T2_only = flag_T2 & ~flag_Q;
is_Q_only = ~flag_T2 & flag_Q;
is_T2_and_Q = flag_T2 & flag_Q; % Consensus outliers
is_OR_outlier = flag_T2 | flag_Q; % OR outliers
is_normal = ~is_OR_outlier; % Normal if neither T2 nor Q outlier

fprintf('Spectra counts: Normal=%d, T2-only=%d, Q-only=%d, T2&Q (Consensus)=%d, Any (T2 or Q)=%d\n', ...
    sum(is_normal),sum(is_T2_only),sum(is_Q_only),sum(is_T2_and_Q), sum(is_OR_outlier));

%% --- 3. Generate Requested Visualizations ---
% (This entire section with PLOT 1 through PLOT 6 remains the same as your current script)
% Make sure the variables used for plotting (is_normal, is_T2_only, is_Q_only, is_T2_and_Q)
% are consistent with the definitions above.
fprintf('\n--- 3. Generating Requested Visualizations ---\n');

% PLOT 1: Q-Statistic / T2 Statistic individual Plots (Revised)
fig1 = figure('Name', 'Individual T2 and Q Statistics'); fig1.Position = [50,500,900,650];
tl1 = tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
sgtitle(tl1,'Individual T2 and Q Statistic Distributions','FontWeight','Normal','FontSize',P.plotFontSize+1);
ax1a=nexttile(tl1);hdl1a=[];leg1a={};
hdl1a(end+1)=plot(ax1a,find(~flag_T2),T2_values(~flag_T2),'o','Color',[0.7 0.7 0.7],'MarkerSize',3);leg1a{end+1}='T2 <= Thresh'; hold(ax1a,'on');
if any(flag_T2),hdl1a(end+1)=plot(ax1a,find(flag_T2),T2_values(flag_T2),'x','Color',P.colorT2OutlierFlag,'MarkerSize',5);leg1a{end+1}='T2 > Thresh';end
h_t2_line=yline(ax1a,T2_threshold,'--','Color',P.colorT2OutlierFlag,'LineWidth',1.5);leg1a{end+1}=sprintf('T2 Thresh=%.2f',T2_threshold);hdl1a(end+1)=h_t2_line;
hold(ax1a,'off');ylabel(ax1a,'T^2 Value');title(ax1a,sprintf('Hotelling T^2 (k_{model}=%d)',k_model),'FontWeight','normal');legend(ax1a,hdl1a,leg1a,'Location','northeast');grid(ax1a,'on');
ax1b=nexttile(tl1);hdl1b=[];leg1b={};
hdl1b(end+1)=plot(ax1b,find(~flag_Q),Q_values(~flag_Q),'o','Color',[0.7 0.7 0.7],'MarkerSize',3);leg1b{end+1}='Q <= Thresh';hold(ax1b,'on');
if any(flag_Q),hdl1b(end+1)=plot(ax1b,find(flag_Q),Q_values(flag_Q),'x','Color',P.colorQOutlierFlag,'MarkerSize',5);leg1b{end+1}='Q > Thresh';end
h_q_line=yline(ax1b,Q_threshold,'--','Color',P.colorQOutlierFlag,'LineWidth',1.5);leg1b{end+1}=sprintf('Q Thresh=%.2g',Q_threshold);hdl1b(end+1)=h_q_line;
hold(ax1b,'off');xlabel(ax1b,'Spectrum Index');ylabel(ax1b,'Q-Statistic');title(ax1b,sprintf('Q-Statistic (SPE) (k_{model}=%d)',k_model),'FontWeight','normal');legend(ax1b,hdl1b,leg1b,'Location','northeast');grid(ax1b,'on');
exportgraphics(fig1,fullfile(figuresDir,sprintf('%s_Vis1_T2_Q_Individual.tiff',P.datePrefix)),'Resolution',300);savefig(fig1,fullfile(figuresDir,sprintf('%s_Vis1_T2_Q_Individual.fig',P.datePrefix)));fprintf('Plot 1 saved.\n');

% PLOT 2: T2 vs Q Plot with Thresholds and Outlier Categories (Revised)
fig2=figure('Name','T2 vs Q with Categories & Labeled Thresholds');fig2.Position=[100 100 800 650];hold on;hdl2=[];leg2={};
if any(is_normal & y_numeric==1),hdl2(end+1)=plot(T2_values(is_normal&y_numeric==1),Q_values(is_normal&y_numeric==1),'o','MarkerSize',4,'MarkerFaceColor',P.colorWHO1,'Color',P.colorWHO1);leg2{end+1}='WHO1(Normal)';end
if any(is_normal & y_numeric==3),hdl2(end+1)=plot(T2_values(is_normal&y_numeric==3),Q_values(is_normal&y_numeric==3),'s','MarkerSize',4,'MarkerFaceColor',P.colorWHO3,'Color',P.colorWHO3);leg2{end+1}='WHO3(Normal)';end
if any(is_T2_only),hdl2(end+1)=plot(T2_values(is_T2_only),Q_values(is_T2_only),'s','MarkerSize',5,'MarkerEdgeColor',P.colorT2OutlierFlag,'MarkerFaceColor',P.colorT2OutlierFlag*0.7);leg2{end+1}='T2-only';end
if any(is_Q_only),hdl2(end+1)=plot(T2_values(is_Q_only),Q_values(is_Q_only),'d','MarkerSize',5,'MarkerEdgeColor',P.colorQOutlierFlag,'MarkerFaceColor',P.colorQOutlierFlag*0.7);leg2{end+1}='Q-only';end
if any(is_T2_and_Q),hdl2(end+1)=plot(T2_values(is_T2_and_Q),Q_values(is_T2_and_Q),'*','MarkerSize',6,'Color',P.colorBothOutlierFlag);leg2{end+1}='T2&Q (Consensus)';end % Updated legend
line([T2_threshold T2_threshold],get(gca,'YLim'),'Color',P.colorT2OutlierFlag,'LineStyle','--','LineWidth',1.2,'HandleVisibility','off');
line(get(gca,'XLim'),[Q_threshold Q_threshold],'Color',P.colorQOutlierFlag,'LineStyle','--','LineWidth',1.2,'HandleVisibility','off');
text(T2_threshold,mean(get(gca,'YLim'))*0.9,sprintf(' T2 Th=%.2f',T2_threshold),'Color',P.colorT2OutlierFlag,'VerticalAlignment','top','HorizontalAlignment','left','FontSize',P.plotFontSize-2,'BackgroundColor','w','EdgeColor','k','Margin',1);
text(mean(get(gca,'XLim')),Q_threshold*1.05,sprintf(' Q Th=%.2g',Q_threshold),'Color',P.colorQOutlierFlag,'VerticalAlignment','bottom','HorizontalAlignment','center','FontSize',P.plotFontSize-2,'BackgroundColor','w','EdgeColor','k','Margin',1);
hold off;xlabel('Hotelling T^2');ylabel('Q-Statistic (SPE)');title(sprintf('T^2 vs. Q (k_{model}=%d)',k_model),'FontWeight','normal');
if ~isempty(hdl2),legend(hdl2,leg2,'Location','NorthEast','FontSize',P.plotFontSize-1);end;grid on;
exportgraphics(fig2,fullfile(figuresDir,sprintf('%s_Vis2_T2vQ_CategoriesAndThresholds.tiff',P.datePrefix)),'Resolution',300);savefig(fig2,fullfile(figuresDir,sprintf('%s_Vis2_T2vQ_CategoriesAndThresholds.fig',P.datePrefix)));fprintf('Plot 2 saved.\n');

% PLOT 3: T2 vs Q Plot only WHO-1 and WHO-3 (Raw Distribution)
fig3 = figure('Name', 'T2 vs Q Raw Distribution by WHO Grade'); fig3.Position = [150 150 750 600]; hold on;
hdl3 = []; leg3 = {}; idx_w1_plot3 = (y_numeric == 1); idx_w3_plot3 = (y_numeric == 3); idx_unk_plot3 = ~(idx_w1_plot3 | idx_w3_plot3);
if any(idx_w1_plot3),hdl3(end+1)=scatter(T2_values(idx_w1_plot3), Q_values(idx_w1_plot3), 15, P.colorWHO1, 'o', 'filled', 'MarkerFaceAlpha',0.4); leg3{end+1}='WHO-1'; end
if any(idx_w3_plot3),hdl3(end+1)=scatter(T2_values(idx_w3_plot3), Q_values(idx_w3_plot3), 15, P.colorWHO3, 's', 'filled', 'MarkerFaceAlpha',0.4); leg3{end+1}='WHO-3'; end
if any(idx_unk_plot3),hdl3(end+1)=scatter(T2_values(idx_unk_plot3), Q_values(idx_unk_plot3), 15, [0.7 0.7 0.7], '^', 'filled', 'MarkerFaceAlpha',0.3); leg3{end+1}='Other/Unknown Grade'; end
hold off; xlabel('Hotelling T^2'); ylabel('Q-Statistic (SPE)'); title(sprintf('T^2 vs. Q Raw Data Distribution (k_{model}=%d)',k_model), 'FontWeight','normal');
if ~isempty(hdl3), legend(hdl3,leg3,'Location','best','FontSize',P.plotFontSize-1); end; grid on;
exportgraphics(fig3, fullfile(figuresDir, sprintf('%s_Vis3_T2vQ_WHOgradesOnly.tiff',P.datePrefix)),'Resolution',300); savefig(fig3, fullfile(figuresDir,sprintf('%s_Vis3_T2vQ_WHOgradesOnly.fig',P.datePrefix))); fprintf('Plot 3 saved.\n');

% PLOT 4a: PCA Scores by WHO Grade Only (2D and 3D in one figure)
fig4a = figure('Name', 'PCA Scores by WHO Grade Only'); fig4a.Position = [50 50 900 450]; tl4a = tiledlayout(1,2,'TileSpacing','compact','Padding','compact'); sgtitle(tl4a,'PCA Scores (Colored by WHO Grade Only)','FontWeight','Normal');
ax4a1=nexttile(tl4a);hold(ax4a1,'on');hdl4a1=[];leg4a1={}; if any(idx_w1_plot3),hdl4a1(end+1)=scatter(ax4a1,score(idx_w1_plot3,1),score(idx_w1_plot3,min(2,size(score,2))),15,P.colorWHO1,'o','filled','MarkerFaceAlpha',0.3);leg4a1{end+1}='WHO-1';end; if any(idx_w3_plot3),hdl4a1(end+1)=scatter(ax4a1,score(idx_w3_plot3,1),score(idx_w3_plot3,min(2,size(score,2))),15,P.colorWHO3,'s','filled','MarkerFaceAlpha',0.3);leg4a1{end+1}='WHO-3';end; if any(idx_unk_plot3),hdl4a1(end+1)=scatter(ax4a1,score(idx_unk_plot3,1),score(idx_unk_plot3,min(2,size(score,2))),15,[0.7 0.7 0.7],'^','filled','MarkerFaceAlpha',0.2);leg4a1{end+1}='Other';end; hold(ax4a1,'off');xlabel(ax4a1,sprintf('PC1(%.1f%%)',explained(1)));ylabel(ax4a1,sprintf('PC2(%.1f%%)',explained(min(2,length(explained)))));title(ax4a1,'2D: PC1 vs PC2','FontWeight','normal');if ~isempty(hdl4a1),legend(ax4a1,hdl4a1,leg4a1,'Location','best','FontSize',P.plotFontSize-2);end;axis(ax4a1,'equal');grid(ax4a1,'on');
ax4a2=nexttile(tl4a);if size(score,2)>=3,hold(ax4a2,'on');hdl4a2=[];leg4a2={}; if any(idx_w1_plot3),hdl4a2(end+1)=scatter3(ax4a2,score(idx_w1_plot3,1),score(idx_w1_plot3,2),score(idx_w1_plot3,3),15,P.colorWHO1,'o','filled','MarkerFaceAlpha',0.3);leg4a2{end+1}='WHO-1';end; if any(idx_w3_plot3),hdl4a2(end+1)=scatter3(ax4a2,score(idx_w3_plot3,1),score(idx_w3_plot3,2),score(idx_w3_plot3,3),15,P.colorWHO3,'s','filled','MarkerFaceAlpha',0.3);leg4a2{end+1}='WHO-3';end; if any(idx_unk_plot3),hdl4a2(end+1)=scatter3(ax4a2,score(idx_unk_plot3,1),score(idx_unk_plot3,2),score(idx_unk_plot3,3),15,[0.7 0.7 0.7],'^','filled','MarkerFaceAlpha',0.2);leg4a2{end+1}='Other';end; hold(ax4a2,'off');view(ax4a2,-30,20);xlabel(ax4a2,sprintf('PC1(%.1f%%)',explained(1)));ylabel(ax4a2,sprintf('PC2(%.1f%%)',explained(2)));zlabel(ax4a2,sprintf('PC3(%.1f%%)',explained(3)));title(ax4a2,'3D: PC1-PC2-PC3','FontWeight','normal');if ~isempty(hdl4a2),legend(ax4a2,hdl4a2,leg4a2,'Location','best','FontSize',P.plotFontSize-2);end;grid(ax4a2,'on');axis(ax4a2,'tight');else,text(0.5,0.5,'<3 PCs','Parent',ax4a2,'HorizontalAlignment','center');title(ax4a2,'3D: PC1-PC2-PC3','FontWeight','normal');end
exportgraphics(fig4a,fullfile(figuresDir,sprintf('%s_Vis4a_PCA_Scores_WHOonly.tiff',P.datePrefix)),'Resolution',300);savefig(fig4a,fullfile(figuresDir,sprintf('%s_Vis4a_PCA_Scores_WHOonly.fig',P.datePrefix)));fprintf('Plot 4a saved.\n');

% PLOT 4b: PCA Scores with Outlier Categories Marked (2D and 3D in one figure)
fig4b = figure('Name','PCA Scores with Outlier Categories');fig4b.Position=[100 50 900 700];tl4b=tiledlayout(2,1,'TileSpacing','compact','Padding','compact');sgtitle(tl4b,'PCA Score Plots (Outlier Categories Marked)','FontWeight','Normal');
ax4b1=nexttile(tl4b);hold(ax4b1,'on');hdl4b1=[];leg4b1={}; if any(is_normal&y_numeric==1),hdl4b1(end+1)=scatter(ax4b1,score(is_normal&y_numeric==1,1),score(is_normal&y_numeric==1,min(2,size(score,2))),15,P.colorWHO1,'o','filled','MarkerFaceAlpha',0.3);leg4b1{end+1}='WHO1(Normal)';end; if any(is_normal&y_numeric==3),hdl4b1(end+1)=scatter(ax4b1,score(is_normal&y_numeric==3,1),score(is_normal&y_numeric==3,min(2,size(score,2))),15,P.colorWHO3,'s','filled','MarkerFaceAlpha',0.3);leg4b1{end+1}='WHO3(Normal)';end; if any(is_T2_only),hdl4b1(end+1)=scatter(ax4b1,score(is_T2_only,1),score(is_T2_only,min(2,size(score,2))),25,P.colorT2OutlierFlag,'s','MarkerFaceColor',P.colorT2OutlierFlag*0.7);leg4b1{end+1}='T2-only';end; if any(is_Q_only),hdl4b1(end+1)=scatter(ax4b1,score(is_Q_only,1),score(is_Q_only,min(2,size(score,2))),25,P.colorQOutlierFlag,'d','MarkerFaceColor',P.colorQOutlierFlag*0.7);leg4b1{end+1}='Q-only';end; if any(is_T2_and_Q),hdl4b1(end+1)=scatter(ax4b1,score(is_T2_and_Q,1),score(is_T2_and_Q,min(2,size(score,2))),30,P.colorBothOutlierFlag,'*');leg4b1{end+1}='T2&Q (Consensus)';end; hold(ax4b1,'off');xlabel(ax4b1,sprintf('PC1(%.1f%%)',explained(1)));ylabel(ax4b1,sprintf('PC2(%.1f%%)',explained(min(2,length(explained)))));title(ax4b1,'2D: PC1 vs PC2','FontWeight','normal');if ~isempty(hdl4b1),legend(ax4b1,hdl4b1,leg4b1,'Location','best','FontSize',P.plotFontSize-2);end;axis(ax4b1,'equal');grid(ax4b1,'on');
ax4b2=nexttile(tl4b);if size(score,2)>=3,hold(ax4b2,'on');hdl4b2=[];leg4b2={}; if any(is_normal&y_numeric==1),hdl4b2(end+1)=scatter3(ax4b2,score(is_normal&y_numeric==1,1),score(is_normal&y_numeric==1,2),score(is_normal&y_numeric==1,3),15,P.colorWHO1,'o','filled','MarkerFaceAlpha',0.3);leg4b2{end+1}='WHO1(Normal)';end; if any(is_normal&y_numeric==3),hdl4b2(end+1)=scatter3(ax4b2,score(is_normal&y_numeric==3,1),score(is_normal&y_numeric==3,2),score(is_normal&y_numeric==3,3),15,P.colorWHO3,'s','filled','MarkerFaceAlpha',0.3);leg4b2{end+1}='WHO3(Normal)';end; if any(is_T2_only),hdl4b2(end+1)=scatter3(ax4b2,score(is_T2_only,1),score(is_T2_only,2),score(is_T2_only,3),25,P.colorT2OutlierFlag,'s','MarkerFaceColor',P.colorT2OutlierFlag*0.7);leg4b2{end+1}='T2-only';end; if any(is_Q_only),hdl4b2(end+1)=scatter3(ax4b2,score(is_Q_only,1),score(is_Q_only,2),score(is_Q_only,3),25,P.colorQOutlierFlag,'d','MarkerFaceColor',P.colorQOutlierFlag*0.7);leg4b2{end+1}='Q-only';end; if any(is_T2_and_Q),hdl4b2(end+1)=scatter3(ax4b2,score(is_T2_and_Q,1),score(is_T2_and_Q,2),score(is_T2_and_Q,3),30,P.colorBothOutlierFlag,'*');leg4b2{end+1}='T2&Q (Consensus)';end; hold(ax4b2,'off');view(ax4b2,-30,20);xlabel(ax4b2,sprintf('PC1(%.1f%%)',explained(1)));ylabel(ax4b2,sprintf('PC2(%.1f%%)',explained(2)));zlabel(ax4b2,sprintf('PC3(%.1f%%)',explained(3)));title(ax4b2,'3D: PC1-PC2-PC3','FontWeight','normal');if ~isempty(hdl4b2),legend(ax4b2,hdl4b2,leg4b2,'Location','best','FontSize',P.plotFontSize-2);end;grid(ax4b2,'on');axis(ax4b2,'tight');else,text(0.5,0.5,'<3 PCs','Parent',ax4b2,'HorizontalAlignment','center');title(ax4b2,'3D: PC1-PC2-PC3','FontWeight','normal');end
exportgraphics(fig4b,fullfile(figuresDir,sprintf('%s_Vis4b_PCA_Scores_OutlierCats.tiff',P.datePrefix)),'Resolution',300);savefig(fig4b,fullfile(figuresDir,sprintf('%s_Vis4b_PCA_Scores_OutlierCats.fig',P.datePrefix)));fprintf('Plot 4b saved.\n');

% PLOT 5: All PC Loadings (PC1 to k_model) in a single tiled layout
num_pcs_for_loadings = k_model;
if num_pcs_for_loadings > 0 && ~isempty(coeff)
    if num_pcs_for_loadings <= 3, ncols_loadings = 1; nrows_loadings = num_pcs_for_loadings;
    elseif num_pcs_for_loadings <= 8, ncols_loadings = 2; nrows_loadings = ceil(num_pcs_for_loadings / ncols_loadings);
    else, ncols_loadings = 3; nrows_loadings = ceil(num_pcs_for_loadings / ncols_loadings); end
    fig5 = figure('Name',sprintf('PCA Loadings (PC1-PC%d of T2/Q Model)',num_pcs_for_loadings));fig5.Position = [100 50 min(ncols_loadings*450,1350) min(nrows_loadings*220,880)];
    tl5 = tiledlayout(nrows_loadings,ncols_loadings,'TileSpacing','compact','Padding','tight'); sgtitle(tl5,sprintf('PCA Loadings for %d PCs used in T2/Q Model',num_pcs_for_loadings),'FontWeight','Normal');
    for pc_idx = 1:num_pcs_for_loadings
        if pc_idx > size(coeff,2), break; end; ax_l = nexttile(tl5); plot(ax_l,wavenumbers_roi,coeff(:,pc_idx),'LineWidth',1); title(ax_l,sprintf('PC%d Loadings (Expl.Var: %.2f%%)',pc_idx,explained(pc_idx)),'FontWeight','normal','FontSize',P.plotFontSize-1); ylabel(ax_l,'Loading Value','FontSize',P.plotFontSize-2);grid(ax_l,'on');set(ax_l,'XDir','reverse','XLim',P.plotXLim,'FontSize',P.plotFontSize-2);
        current_tile_info=get(ax_l,'Layout');current_col=mod(current_tile_info.Tile-1,ncols_loadings)+1;current_row=ceil(current_tile_info.Tile/ncols_loadings);
        if current_col~=1,set(ax_l,'YTickLabel',[]);end;if current_row~=nrows_loadings,set(ax_l,'XTickLabel',[]);end
    end; xlabel(tl5,P.plotXLabel,'FontSize',P.plotFontSize-1);
    exportgraphics(fig5,fullfile(figuresDir,sprintf('%s_Vis5_PCA_Loadings_kModel.tiff',P.datePrefix)),'Resolution',300);savefig(fig5,fullfile(figuresDir,sprintf('%s_Vis5_PCA_Loadings_kModel.fig',P.datePrefix)));fprintf('Plot 5 saved.\n');
else, fprintf('Skipping Plot 5 (Loadings) as k_model is 0 or coeffs empty.\n'); end

% PLOT 6: Tiled Layout of Spectra for Distinct Outlier Categories (Revised Mean Line)
fig6 = figure('Name', 'Spectra of Distinct Outlier Categories'); fig6.Position = [120 120 700 850];
tl6 = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
sgtitle(tl6,'Spectra by Distinct Outlier Category','FontWeight','Normal', 'FontSize', P.plotFontSize+1);
outlier_cats_plot6 = {{'Q-only Flagged',is_Q_only,P.colorQOutlierFlag},{'T2-only Flagged',is_T2_only,P.colorT2OutlierFlag},{'T2&Q Flagged (Consensus)',is_T2_and_Q,P.colorBothOutlierFlag}};
for cat_idx=1:3, ax_cat=nexttile(tl6);hold(ax_cat,'on');cat_title_base=outlier_cats_plot6{cat_idx}{1};cat_flag=outlier_cats_plot6{cat_idx}{2};cat_color_lines=outlier_cats_plot6{cat_idx}{3};
    spectra_cat=X(cat_flag,:);num_cat=sum(cat_flag);hdl_cat_mean=[];
    if num_cat>0
        plot(ax_cat,wavenumbers_roi,spectra_cat','Color',[cat_color_lines,0.1],'LineWidth',0.5,'HandleVisibility','off');
        mean_spec=mean(spectra_cat,1,'omitnan');
        if any(~isnan(mean_spec)),hdl_cat_mean=plot(ax_cat,wavenumbers_roi,mean_spec,'Color','k','LineWidth',1.5,'DisplayName',sprintf('Mean (n=%d)',num_cat)); uistack(hdl_cat_mean,'top'); end
    else, text(0.5,0.5,'No spectra','Parent',ax_cat,'HorizontalAlignment','center');end
    hold(ax_cat,'off');title(ax_cat,sprintf('%s (n=%d)',cat_title_base,num_cat),'FontWeight','normal');ylabel(ax_cat,P.plotYLabelAbsorption);set(ax_cat,'XDir','reverse','XLim',P.plotXLim,'FontSize',P.plotFontSize-1);grid(ax_cat,'on');
    if cat_idx<3,set(ax_cat,'XTickLabel',[]);else,xlabel(ax_cat,P.plotXLabel);end
    if ~isempty(hdl_cat_mean) && isgraphics(hdl_cat_mean),legend(ax_cat,hdl_cat_mean,'Location','northeast','FontSize',P.plotFontSize-2);end
end
drawnow; pause(0.2);
tiffFile6 = fullfile(figuresDir, sprintf('%s_Vis6_OutlierCategory_Spectra.tiff', P.datePrefix));
figFile6 = fullfile(figuresDir, sprintf('%s_Vis6_OutlierCategory_Spectra.fig', P.datePrefix));
if isvalid(fig6) && isgraphics(fig6,'figure')
    fprintf('fig6 is valid before attempting to save.\n');
    try, exportgraphics(fig6, tiffFile6, 'Resolution', 300); fprintf('Plot 6 TIFF saved successfully using exportgraphics.\n');
    catch ME_export, fprintf('WARNING: exportgraphics failed for Plot 6 TIFF: %s\nAttempting print command...\n', ME_export.message);
        if isvalid(fig6) && isgraphics(fig6,'figure')
            try, print(fig6, tiffFile6, '-dtiff', '-r300'); fprintf('Plot 6 TIFF saved successfully using print command.\n');
            catch ME_print, fprintf('ERROR: print command also failed for Plot 6 TIFF: %s\n', ME_print.message); end
        else, warning('Plot 6 (fig6) became invalid before print could be attempted.');end
    end
    if isvalid(fig6) && isgraphics(fig6,'figure')
        fprintf('fig6 is still valid before savefig.\n');
        try, savefig(fig6, figFile6); fprintf('Plot 6 FIG saved successfully.\n');
        catch ME_savefig, fprintf('ERROR saving Plot 6 FIG: %s\n', ME_savefig.message);end
    else, warning('Plot 6 (fig6) became invalid before savefig for .fig file.');end
else, warning('Plot 6 figure handle (fig6) was invalid BEFORE attempting to save.');end
fprintf('All requested diagnostic visualizations generated.\n');


%% --- 4. Save Exploratory Analysis Data ---
fprintf('\n--- 4. Saving Exploratory Analysis Data ---\n');
% This data is crucial for the separate outlier removal script.
exploratoryOutlierData = struct();
exploratoryOutlierData.scriptRunDate = P.datePrefix;
exploratoryOutlierData.dataSource = 'dataTableTrain.mat (or equivalent structure)'; % Note source
exploratoryOutlierData.alpha_T2_Q = P.alpha_T2_Q; % Store the alpha used
exploratoryOutlierData.variance_to_explain_for_PCA_model = P.variance_to_explain_for_PCA_model;
exploratoryOutlierData.k_model = k_model;

% T2/Q values and flags for ALL spectra in the analyzed X
exploratoryOutlierData.T2_values_all_spectra = T2_values;
exploratoryOutlierData.T2_threshold = T2_threshold;
exploratoryOutlierData.flag_T2_outlier_all_spectra = flag_T2;

exploratoryOutlierData.Q_values_all_spectra = Q_values;
exploratoryOutlierData.Q_threshold = Q_threshold;
exploratoryOutlierData.flag_Q_outlier_all_spectra = flag_Q;

% Combined flags for convenience (still for ALL spectra)
exploratoryOutlierData.flag_T2_only_all_spectra = is_T2_only;
exploratoryOutlierData.flag_Q_only_all_spectra = is_Q_only;
exploratoryOutlierData.flag_T2_and_Q_all_spectra = is_T2_and_Q; % Consensus
exploratoryOutlierData.flag_OR_outlier_all_spectra = is_OR_outlier;
exploratoryOutlierData.flag_Normal_all_spectra = is_normal;

% Mapping information back to original dataTableTrain structure
exploratoryOutlierData.Original_ProbeRowIndices_map = Original_ProbeRowIndices;
exploratoryOutlierData.Original_SpectrumIndexInProbe_map = Original_SpectrumIndexInProbe;
exploratoryOutlierData.Patient_ID_map = Patient_ID; % For all spectra in X
exploratoryOutlierData.y_numeric_map = y_numeric;   % For all spectra in X
exploratoryOutlierData.y_categorical_map = y_cat; % For all spectra in X

% PCA model from this run (on the full X)
exploratoryOutlierData.PCA_coeff = coeff;
exploratoryOutlierData.PCA_mu = mu;
exploratoryOutlierData.PCA_latent = latent;
exploratoryOutlierData.PCA_explained = explained;
exploratoryOutlierData.PCA_scores_all_spectra = score; % Scores for ALL spectra

exploratoryFilename_mat = fullfile(resultsDir, sprintf('%s_ExploratoryOutlier_AnalysisData.mat', P.datePrefix));
save(exploratoryFilename_mat, 'exploratoryOutlierData', '-v7.3');
fprintf('Exploratory outlier ANALYSIS DATA (T2/Q values, flags, PCA model for ALL spectra) saved to: %s\n', exploratoryFilename_mat);
fprintf('This .mat file is intended as input for a separate outlier REMOVAL script.\n');

fprintf('\n--- Focused Exploratory Outlier Visualization & Analysis Data Generation Script Finished ---\n');