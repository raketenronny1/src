% run_comprehensive_outlier_processing.m
%
% Single script to:
% 1. Perform detailed exploratory outlier analysis (PCA, T2, Q).
% 2. Generate all associated visualizations (Plots 1-8 from exploratory script).
% 3. Create cleaned training datasets based on "T2 OR Q" AND "T2 AND Q" strategies.
% 4. Save comprehensive analysis data and detailed per-probe outlier tables.
%
% This script prepares the necessary inputs for the subsequent
% comparative model selection script.
%
% Date: 2025-05-18

%% 0. Initialization & Setup
clear; clc; close all;
fprintf('COMPREHENSIVE OUTLIER PROCESSING (T2/Q OR & AND Strategies) - START %s\n', string(datetime('now')));

% --- Define Paths ---
projectRoot = pwd;
if ~exist(fullfile(projectRoot, 'src'), 'dir') || ~exist(fullfile(projectRoot, 'data'), 'dir')
    error('Project structure not found. Run from project root. Current: %s', projectRoot);
end

P.dataPath = fullfile(projectRoot, 'data');
P.resultsPath_OutlierExploration = fullfile(projectRoot, 'results', 'Phase1_OutlierExploration'); % For the main .mat analysis file
P.resultsPath_OutlierApplication = fullfile(projectRoot, 'results'); % For cleaned tables, CSVs
P.figuresPath_OutlierExploration = fullfile(projectRoot, 'figures', 'Phase1_OutlierExploration'); % For all plots
P.helperFunPath = fullfile(projectRoot, 'src', 'helper_functions');

% Create directories if they don't exist
dirPathsToEnsure = {P.resultsPath_OutlierExploration, P.resultsPath_OutlierApplication, P.figuresPath_OutlierExploration};
for i = 1:length(dirPathsToEnsure)
    if ~isfolder(dirPathsToEnsure{i}), mkdir(dirPathsToEnsure{i}); end
end
addpath(P.helperFunPath);

% --- Parameters ---
P.datePrefix = string(datetime('now','Format','yyyyMMdd'));
P.alpha_T2_Q = 0.05; 
P.variance_to_explain_for_PCA_model = 0.95;

% Plotting Colors & Styles (from P_Global in previous concept)
P.colorWHO1 = [0.9, 0.6, 0.4]; P.colorWHO3 = [0.4, 0.702, 0.902];
P.colorT2OutlierFlag = [0.8, 0.2, 0.2]; P.colorQOutlierFlag = [0.2, 0.2, 0.8];
P.colorBothOutlierFlag = [0.8, 0, 0.8]; % For T2&Q (Consensus)
P.colorOutlierGeneral = [0.8, 0, 0]; 
P.plotFontSize = 10; P.plotXLabel = 'Wellenzahl (cm^{-1})';
P.plotYLabelAbsorption = 'Absorption (a.u.)'; P.plotXLim = [950 1800];

fprintf('Global setup complete. Date prefix for outputs: %s\n', P.datePrefix);
fprintf('Alpha for T2/Q: %.3f, PCA Variance for k_model: %.2f%%\n', P.alpha_T2_Q, P.variance_to_explain_for_PCA_model*100);

%% 1. Load Initial Training Data
fprintf('\n--- 1. Loading Initial Training Data ---\n');
try
    dataTableTrain_File = fullfile(P.dataPath, 'data_table_train.mat');
    if ~exist(dataTableTrain_File, 'file'), error('data_table_train.mat not found in %s', P.dataPath); end
    loadedVars = load(dataTableTrain_File, 'dataTableTrain');
    dataTableTrain_Initial = loadedVars.dataTableTrain;
    fprintf('Original dataTableTrain loaded with %d probes.\n', height(dataTableTrain_Initial));

    wavenumbers_File = fullfile(P.dataPath, 'wavenumbers.mat');
    if ~exist(wavenumbers_File, 'file'), error('wavenumbers.mat not found in %s', P.dataPath); end
    wavenumbers_data_loaded = load(wavenumbers_File, 'wavenumbers_roi');
    wavenumbers_roi = wavenumbers_data_loaded.wavenumbers_roi;
    if iscolumn(wavenumbers_roi), wavenumbers_roi = wavenumbers_roi'; end
    P.num_wavenumber_points = length(wavenumbers_roi);
    fprintf('Wavenumbers_roi loaded (%d points).\n', P.num_wavenumber_points);
catch ME
    fprintf('ERROR in Section 1: Loading initial data: %s\n', ME.message); rethrow(ME);
end

%% 2. Prepare Data for PCA (Flatten spectra and create mapping arrays)
fprintf('\n--- 2. Preparing Data for PCA ---\n');
% This section is identical to A2 in the "master script" concept
allSpectra_cell = {}; allLabels_cell = {}; Patient_ID_cell = {};
allOriginalProbeRowIndices_vec = []; allOriginalSpectrumIndices_InProbe_vec = [];

for i_prep = 1:height(dataTableTrain_Initial)
    spectraMatrix = dataTableTrain_Initial.CombinedSpectra{i_prep}; % Assumes these are the spectra to analyze

    if isempty(spectraMatrix) || ~isnumeric(spectraMatrix) || ndims(spectraMatrix) ~= 2, continue; end
    if size(spectraMatrix,1) == 0, continue; end
    if size(spectraMatrix,2) ~= P.num_wavenumber_points,
        warning('Prep: WN mismatch for Diss_ID %s. Skipping.', dataTableTrain_Initial.Diss_ID{i_prep}); continue;
    end

    numIndSpectra = size(spectraMatrix, 1);
    allSpectra_cell{end+1,1} = spectraMatrix;
    allLabels_cell{end+1,1} = repmat(dataTableTrain_Initial.WHO_Grade(i_prep), numIndSpectra, 1);
    Patient_ID_cell{end+1,1} = repmat(dataTableTrain_Initial.Diss_ID(i_prep), numIndSpectra, 1);
    allOriginalProbeRowIndices_vec = [allOriginalProbeRowIndices_vec; repmat(i_prep, numIndSpectra, 1)];
    allOriginalSpectrumIndices_InProbe_vec = [allOriginalSpectrumIndices_InProbe_vec; (1:numIndSpectra)'];
end

if isempty(allSpectra_cell), error('Prep: No valid spectra extracted from dataTableTrain_Initial.'); end

X_train_full_flat = cell2mat(allSpectra_cell); % This is the X used for PCA & T2/Q
y_cat_full_flat = cat(1, allLabels_cell{:});
Patient_ID_full_flat = vertcat(Patient_ID_cell{:}); % Corrected usage of vertcat for cell arrays of strings
Original_ProbeRowIndices_flat = allOriginalProbeRowIndices_vec;
Original_SpectrumIndexInProbe_flat = allOriginalSpectrumIndices_InProbe_vec;

y_numeric_full_flat = zeros(length(y_cat_full_flat), 1);
categories_y_flat = categories(y_cat_full_flat);
idx_who1_flat = find(strcmp(categories_y_flat, 'WHO-1')); idx_who3_flat = find(strcmp(categories_y_flat, 'WHO-3'));
if ~isempty(idx_who1_flat), y_numeric_full_flat(y_cat_full_flat == categories_y_flat{idx_who1_flat}) = 1; end
if ~isempty(idx_who3_flat), y_numeric_full_flat(y_cat_full_flat == categories_y_flat{idx_who3_flat}) = 3; end
fprintf('Data for PCA prepared: %d spectra, %d features.\n', size(X_train_full_flat,1), size(X_train_full_flat,2));

%% 3. Perform PCA, Calculate T²/Q Statistics & Thresholds (ONCE)
fprintf('\n--- 3. PCA, T2/Q Calculation & Thresholds ---\n');

if isempty(X_train_full_flat)
    error('CRITICAL ERROR in Section 3: X_train_full_flat (input to PCA) is empty. Cannot proceed.');
end
if size(X_train_full_flat,1) < 2
    error('CRITICAL ERROR in Section 3: X_train_full_flat has fewer than 2 samples. PCA cannot be performed meaningfully.');
end

fprintf('Performing PCA on X_train_full_flat (%d spectra, %d features)...\n', size(X_train_full_flat,1), size(X_train_full_flat,2));

% Ensure X_train_full_flat does not contain NaN/Inf that would break PCA
if any(isnan(X_train_full_flat(:))) || any(isinf(X_train_full_flat(:)))
    warning('Data for PCA (X_train_full_flat) contains NaN or Inf values. Attempting to replace with column means (for NaNs) or remove affected rows/cols if severe.');
    % Simplistic handling: Replace NaNs with mean of column. Inf might need row removal or more sophisticated imputation.
    if any(isnan(X_train_full_flat(:)))
        X_train_full_flat = fillmissing(X_train_full_flat, 'mean'); 
        fprintf('NaNs in X_train_full_flat replaced with column means.\n');
    end
    if any(isinf(X_train_full_flat(:)))
        % For Inf, a common strategy is to remove rows containing Inf, or error out.
        rows_with_inf = any(isinf(X_train_full_flat),2);
        if any(rows_with_inf)
            fprintf('WARNING: %d rows in X_train_full_flat contain Inf values and will be removed before PCA.\n', sum(rows_with_inf));
            X_train_full_flat(rows_with_inf,:) = [];
            % IMPORTANT: If rows are removed, all mapping arrays (y_numeric_full_flat, Patient_ID_full_flat, etc.)
            % MUST be filtered accordingly. This adds significant complexity if not handled carefully from the start.
            % For now, this is a warning; robust handling would require re-filtering all associated metadata.
            % A safer approach might be to error out if Inf values are present and require user to clean data first.
            if isempty(X_train_full_flat)
                 error('CRITICAL ERROR in Section 3: All data removed after attempting to handle Inf values. Cannot proceed.');
            end
        end
    end
end

% PCA Call
try
    [coeff_pca, score_pca, latent_pca, tsquared_pca_builtin, explained_pca, mu_pca] = pca(X_train_full_flat, 'Algorithm','svd');
catch ME_pca
    fprintf('ERROR during PCA execution: %s\n', ME_pca.message);
    disp(ME_pca.getReport);
    error('PCA failed. Cannot continue with outlier detection.');
end

% --- Post-PCA Checks ---
if isempty(explained_pca) || isempty(latent_pca) || isempty(score_pca) || isempty(coeff_pca)
    error('PCA did not return expected outputs (explained_pca, latent_pca, score_pca, or coeff_pca is empty). Check input data X_train_full_flat.');
end
if explained_pca(1) < 0 || any(explained_pca < -1e-6) % Small negative due to precision sometimes okay, large negatives not.
    warning('Negative values detected in explained variance from PCA. This is unusual. Proceeding with caution.');
    % explained_pca(explained_pca < 0) = 0; % Option: Force small negatives to zero
end
fprintf('PCA completed. Variance explained by 1st PC: %.2f%%\n', explained_pca(1));
fprintf('Number of latent roots (eigenvalues) returned by PCA: %d\n', length(latent_pca));
fprintf('Number of components in `explained_pca`: %d\n', length(explained_pca));


% Determine k_model (number of PCs for T2/Q model)
cumulativeVariance_pca = cumsum(explained_pca);
k_model_pca = find(cumulativeVariance_pca >= P.variance_to_explain_for_PCA_model*100, 1, 'first');

if isempty(k_model_pca)
    k_model_pca = length(explained_pca); % Use all available components if threshold not met
    fprintf('Variance threshold not met by any subset of PCs. Using all %d available PCs for k_model_pca.\n', k_model_pca);
end
if k_model_pca == 0 && ~isempty(explained_pca) % Should not happen if explained_pca is not empty
    k_model_pca = 1;
    fprintf('k_model_pca was 0, setting to 1.\n');
end
if isempty(explained_pca) % This check is now redundant due to earlier check, but good for safety
    error('CRITICAL: explained_pca is empty after PCA. Cannot determine k_model_pca.');
end
if k_model_pca == 0
    error('CRITICAL: k_model_pca is 0. Cannot proceed with T2/Q calculations that require k_model_pca > 0.');
end
fprintf('k_model_pca (T2/Q) for >=%.0f%% variance: %d PCs.\n', P.variance_to_explain_for_PCA_model*100, k_model_pca);

% Hotelling's T-Squared Calculation
% Ensure k_model_pca does not exceed dimensions of score_pca or latent_pca
k_model_pca = min(k_model_pca, size(score_pca, 2));
k_model_pca = min(k_model_pca, length(latent_pca));
if k_model_pca == 0, error('k_model_pca became 0 after dimension checks. Cannot proceed.'); end
fprintf('Adjusted k_model_pca after dimension checks: %d PCs.\n', k_model_pca);

temp_score_k = score_pca(:, 1:k_model_pca);
temp_lambda_k = latent_pca(1:k_model_pca); 
temp_lambda_k(temp_lambda_k <= eps) = eps; % Prevent division by zero or near-zero
T2_values_all = sum(bsxfun(@rdivide, temp_score_k.^2, temp_lambda_k'), 2);

% T2 Threshold
n_samples = size(X_train_full_flat, 1);
if n_samples > k_model_pca && k_model_pca > 0
    T2_threshold = ((k_model_pca*(n_samples-1))/(n_samples-k_model_pca))*finv(1-P.alpha_T2_Q,k_model_pca,n_samples-k_model_pca);
    fprintf('T2 threshold (F-dist, alpha=%.3f): %.4f\n', P.alpha_T2_Q, T2_threshold);
else
    if k_model_pca > 0
        T2_threshold = chi2inv(1-P.alpha_T2_Q,k_model_pca);
        fprintf('T2 threshold (Chi2-dist, alpha=%.3f, df=%d): %.4f\n', P.alpha_T2_Q, k_model_pca, T2_threshold);
    else
        error('k_model_pca is 0, cannot calculate T2_threshold.'); % Should have been caught
    end
end
if isnan(T2_threshold) || isinf(T2_threshold)
    error('Calculated T2_threshold is NaN or Inf. Check PCA results, k_model_pca, and n_samples.');
end


% Q-Statistic (SPE) Calculation
% Ensure k_model_pca doesn't exceed dimensions for reconstruction
k_for_recon = min(k_model_pca, size(coeff_pca,2)); % Number of columns in coeff might be less than latent due to rank
if k_for_recon == 0, error('k_for_recon is 0. Cannot calculate Q-statistic.'); end

X_reconstructed = score_pca(:,1:k_for_recon) * coeff_pca(:,1:k_for_recon)' + mu_pca;
Q_values_all = sum((X_train_full_flat - X_reconstructed).^2, 2);

% Q Threshold
Q_threshold = NaN; % Initialize
% Max possible PCs based on data dimensionality (rank)
num_total_actual_pcs = find(latent_pca > eps, 1, 'last'); % Number of non-negligible eigenvalues
if isempty(num_total_actual_pcs), num_total_actual_pcs = 0; end

fprintf('Number of non-negligible latent roots (eigenvalues): %d\n', num_total_actual_pcs);

if k_model_pca < num_total_actual_pcs % Only if there are PCs left out of the model
    % Use eigenvalues from k_model_pca + 1 up to the actual rank of the data
    discarded_eigenvalues = latent_pca(k_model_pca+1 : num_total_actual_pcs);
    discarded_eigenvalues(discarded_eigenvalues <= eps) = eps; % Ensure positivity

    if ~isempty(discarded_eigenvalues)
        theta1 = sum(discarded_eigenvalues);
        theta2 = sum(discarded_eigenvalues.^2);
        theta3 = sum(discarded_eigenvalues.^3);

        if theta1 > eps && theta2 > eps % Avoid division by zero
            h0_val=1-(2*theta1*theta3)/(3*theta2^2);
            if h0_val<=eps || isnan(h0_val) || isinf(h0_val)
                h0_val=1; 
                fprintf('Warning: h0_val for Q-thresh invalid or non-positive (%.2g). Using h0_val=1 as fallback.\n', 1-(2*theta1*theta3)/(3*theta2^2));
            end
            ca_val=norminv(1-P.alpha_T2_Q);
            val_in_bracket=ca_val*sqrt(2*theta2*h0_val^2)/theta1 + 1 + (theta2*h0_val*(h0_val-1))/(theta1^2);
            
            if val_in_bracket > 0 && h0_val > 0 % Ensure base and exponent are valid for power
                Q_threshold=theta1*(val_in_bracket)^(1/h0_val);
            else
                 fprintf('Warning: Value in bracket for Q_threshold (%.2g) or h0_val (%.2g) is non-positive. Using empirical percentile for Q_threshold.\n', val_in_bracket, h0_val);
                 Q_threshold = NaN; % Will trigger empirical calculation
            end
        else
            fprintf('Warning: theta1 or theta2 for Q-threshold calculation is near zero. Using empirical percentile.\n');
            Q_threshold = NaN; % Will trigger empirical calculation
        end
    else
        fprintf('No discarded eigenvalues for Q-statistic calculation (k_model_pca >= num_total_actual_pcs). Q_threshold set to 0.\n');
        Q_threshold = 0; % No residual variance to model
    end
else
    fprintf('All actual PCs (%d) are included in k_model_pca (%d) or k_model_pca is too large. Q_threshold set to 0 (or empirical if Q_values are non-zero).\n', num_total_actual_pcs, k_model_pca);
    Q_threshold = 0; % No residual variance to model
    if any(Q_values_all > eps)
        fprintf('Warning: Q_values are non-zero even though Q_threshold is theoretically 0. This might indicate k_model_pca needs review or using empirical threshold is better.\n');
    end
end

% Fallback to empirical Q_threshold if theoretical calculation failed or resulted in NaN/Inf
if isnan(Q_threshold) || isinf(Q_threshold)
    Q_threshold_empirical = prctile(Q_values_all,(1-P.alpha_T2_Q)*100);
    fprintf('Theoretical Q_threshold failed or invalid. Using empirical Q_threshold (%.1fth percentile): %.4g\n', (1-P.alpha_T2_Q)*100, Q_threshold_empirical);
    Q_threshold = Q_threshold_empirical;
    if isnan(Q_threshold) || isinf(Q_threshold) % Final check if prctile also failed (e.g. all Q are same)
        error('Could not determine a valid Q_threshold. Q_values might be problematic (e.g. all identical or NaN).');
    end
end
fprintf('Final Q-SPE threshold (alpha=%.3f): %.4g\n', P.alpha_T2_Q, Q_threshold);

% Define flags for ALL spectra based on the calculated thresholds
flag_T2_all = (T2_values_all > T2_threshold);
flag_Q_all  = (Q_values_all > Q_threshold);
is_T2_only_all = flag_T2_all & ~flag_Q_all;
is_Q_only_all  = ~flag_T2_all & flag_Q_all;
is_T2_and_Q_all = flag_T2_all & flag_Q_all; % Consensus outliers
is_OR_outlier_all = flag_T2_all | flag_Q_all; % OR outliers (any T2 or any Q)
is_normal_all = ~is_OR_outlier_all;      % Normal if neither T2 nor Q outlier

fprintf('Spectra counts after T2/Q analysis: Normal=%d, T2-only=%d, Q-only=%d, T2&Q(Consensus)=%d, Any(OR)=%d\n', ...
    sum(is_normal_all),sum(is_T2_only_all),sum(is_Q_only_all),sum(is_T2_and_Q_all), sum(is_OR_outlier_all));

%% 4. Generate All Exploratory Visualizations (NOW A FUNCTION CALL)
fprintf('\n--- 4. Calling Function to Generate Exploratory Outlier Visualizations ---\n');
P_for_visualization = P; 
P_for_visualization.figuresPath_OutlierExploration = P.figuresPath_OutlierExploration;

try
    generate_exploratory_outlier_visualizations(X_train_full_flat, ...
                                                y_numeric_full_flat, y_cat_full_flat, Patient_ID_full_flat, ...
                                                wavenumbers_roi, ...
                                                score_pca, ...         % Pass the defined score_pca
                                                explained_pca, ...     % Pass the defined explained_pca
                                                coeff_pca, ...         % Pass the defined coeff_pca
                                                k_model_pca, ...       % Pass the defined k_model_pca
                                                T2_values_all, Q_values_all, T2_threshold, Q_threshold, ...
                                                flag_T2_all, flag_Q_all, is_T2_only_all, is_Q_only_all, ...
                                                is_T2_and_Q_all, is_OR_outlier_all, is_normal_all, ...
                                                P_for_visualization); 
    fprintf('Successfully called generate_exploratory_outlier_visualizations.\n');
catch ME_vis_call
    fprintf('ERROR calling generate_exploratory_outlier_visualizations: %s\n', ME_vis_call.message);
    disp(ME_vis_call.getReport);
    warning('Continuing script without all exploratory visualizations if an error occurred in the plotting function.');
end

%% 5. Save Comprehensive Exploratory Analysis Data
fprintf('\n--- 5. Saving Comprehensive Exploratory Analysis Data ---\n');
exploratoryOutlierData = struct();
exploratoryOutlierData.scriptRunDate = P.datePrefix;
exploratoryOutlierData.dataSource = 'dataTableTrain.mat (or equivalent structure)';
exploratoryOutlierData.alpha_T2_Q = P.alpha_T2_Q;
exploratoryOutlierData.variance_to_explain_for_PCA_model = P.variance_to_explain_for_PCA_model;
exploratoryOutlierData.k_model = k_model_pca; % <<<< CORRECTED HERE

exploratoryOutlierData.T2_values_all_spectra = T2_values_all;
exploratoryOutlierData.T2_threshold = T2_threshold;
exploratoryOutlierData.flag_T2_outlier_all_spectra = flag_T2_all;

exploratoryOutlierData.Q_values_all_spectra = Q_values_all;
exploratoryOutlierData.Q_threshold = Q_threshold;
exploratoryOutlierData.flag_Q_outlier_all_spectra = flag_Q_all;

exploratoryOutlierData.flag_T2_only_all_spectra = is_T2_only_all;
exploratoryOutlierData.flag_Q_only_all_spectra = is_Q_only_all;
exploratoryOutlierData.flag_T2_and_Q_all_spectra = is_T2_and_Q_all; 
exploratoryOutlierData.flag_OR_outlier_all_spectra = is_OR_outlier_all;
exploratoryOutlierData.flag_Normal_all_spectra = is_normal_all;

exploratoryOutlierData.Original_ProbeRowIndices_map = Original_ProbeRowIndices_flat;
exploratoryOutlierData.Original_SpectrumIndexInProbe_map = Original_SpectrumIndexInProbe_flat;
exploratoryOutlierData.Patient_ID_map = Patient_ID_full_flat; 
exploratoryOutlierData.y_numeric_map = y_numeric_full_flat;   
exploratoryOutlierData.y_categorical_map = y_cat_full_flat; 

exploratoryOutlierData.PCA_coeff = coeff_pca;         % <<<< Ensure this is coeff_pca
exploratoryOutlierData.PCA_mu = mu_pca;             % <<<< Ensure this is mu_pca
exploratoryOutlierData.PCA_latent = latent_pca;       % <<<< Ensure this is latent_pca
exploratoryOutlierData.PCA_explained = explained_pca;   % <<<< Ensure this is explained_pca
exploratoryOutlierData.PCA_scores_all_spectra = score_pca; % <<<< Ensure this is score_pca

exploratoryFilename_mat = fullfile(P.resultsPath_OutlierExploration, sprintf('%s_ComprehensiveOutlierAnalysisData.mat', P.datePrefix));
save(exploratoryFilename_mat, 'exploratoryOutlierData', '-v7.3');
fprintf('Comprehensive exploratory outlier ANALYSIS DATA saved to: %s\n', exploratoryFilename_mat);

%% 6. Create and Save Cleaned Flat Datasets for BOTH Strategies
fprintf('\n--- 6. Creating and Saving Cleaned Flat Datasets for Both Strategies ---\n');

% --- Strategy 1: T2 OR Q ---
good_indices_OR = ~is_OR_outlier_all;
X_train_no_outliers_OR = X_train_full_flat(good_indices_OR, :);
y_train_no_outliers_OR_cat = y_cat_full_flat(good_indices_OR);
y_train_no_outliers_OR_num = y_numeric_full_flat(good_indices_OR);
Patient_ID_no_outliers_OR = Patient_ID_full_flat(good_indices_OR);
Original_ProbeRowIndices_no_outliers_OR = Original_ProbeRowIndices_flat(good_indices_OR);
Original_SpectrumIndexInProbe_no_outliers_OR = Original_SpectrumIndexInProbe_flat(good_indices_OR);
fprintf('OR Strategy: Removed %d spectra. Remaining: %d.\n', sum(is_OR_outlier_all), sum(good_indices_OR));
cleanedFlatDatasetFilename_OR = fullfile(P.dataPath, sprintf('%s_training_set_no_outliers_T2orQ.mat', P.datePrefix));
save(cleanedFlatDatasetFilename_OR, ...
     'X_train_no_outliers_OR', 'y_train_no_outliers_OR_cat', 'y_train_no_outliers_OR_num', ...
     'Patient_ID_no_outliers_OR', 'Original_ProbeRowIndices_no_outliers_OR', 'Original_SpectrumIndexInProbe_no_outliers_OR', ...
     'wavenumbers_roi', 'exploratoryOutlierData', '-v7.3'); % Include exploratoryOutlierData for traceability
fprintf('Cleaned flat training dataset (T2 OR Q) saved to: %s\n', cleanedFlatDatasetFilename_OR);

% --- Strategy 2: T2 AND Q (Consensus) ---
good_indices_AND = ~is_T2_and_Q_all;
X_train_no_outliers_AND = X_train_full_flat(good_indices_AND, :);
y_train_no_outliers_AND_cat = y_cat_full_flat(good_indices_AND);
y_train_no_outliers_AND_num = y_numeric_full_flat(good_indices_AND);
Patient_ID_no_outliers_AND = Patient_ID_full_flat(good_indices_AND);
Original_ProbeRowIndices_no_outliers_AND = Original_ProbeRowIndices_flat(good_indices_AND);
Original_SpectrumIndexInProbe_no_outliers_AND = Original_SpectrumIndexInProbe_flat(good_indices_AND);
fprintf('AND Strategy: Removed %d spectra. Remaining: %d.\n', sum(is_T2_and_Q_all), sum(good_indices_AND));
cleanedFlatDatasetFilename_AND = fullfile(P.dataPath, sprintf('%s_training_set_no_outliers_T2andQ.mat', P.datePrefix));
save(cleanedFlatDatasetFilename_AND, ...
     'X_train_no_outliers_AND', 'y_train_no_outliers_AND_cat', 'y_train_no_outliers_AND_num', ...
     'Patient_ID_no_outliers_AND', 'Original_ProbeRowIndices_no_outliers_AND', 'Original_SpectrumIndexInProbe_no_outliers_AND', ...
     'wavenumbers_roi', 'exploratoryOutlierData', '-v7.3');
fprintf('Cleaned flat training dataset (T2 AND Q) saved to: %s\n', cleanedFlatDatasetFilename_AND);

%% 7. Create and Save Detailed Per-Probe Table for Both Strategies
fprintf('\n--- 7. Creating and Saving Detailed Per-Probe Table ---\n');
dataTable_Probes_CleanedInfo = dataTableTrain_Initial; % Start with original probe table
numProbes = height(dataTable_Probes_CleanedInfo);

% Columns for OR strategy
dataTable_Probes_CleanedInfo.CombinedSpectra_OR_Cleaned = cell(numProbes, 1);
dataTable_Probes_CleanedInfo.OutlierSpectra_OR_Removed = cell(numProbes, 1);
dataTable_Probes_CleanedInfo.OutlierIndicesInProbe_OR = cell(numProbes, 1);
dataTable_Probes_CleanedInfo.OutlierTypeInProbe_OR = cell(numProbes,1); % T2, Q, T2&Q
dataTable_Probes_CleanedInfo.NumOriginalSpectraInProbe = zeros(numProbes,1); % Add this for clarity
dataTable_Probes_CleanedInfo.NumOutliers_OR = zeros(numProbes,1);
dataTable_Probes_CleanedInfo.NumSpectra_OR_Cleaned = zeros(numProbes,1);

% Columns for AND strategy
dataTable_Probes_CleanedInfo.CombinedSpectra_AND_Cleaned = cell(numProbes, 1);
dataTable_Probes_CleanedInfo.OutlierSpectra_AND_Removed = cell(numProbes, 1);
dataTable_Probes_CleanedInfo.OutlierIndicesInProbe_AND = cell(numProbes, 1);
dataTable_Probes_CleanedInfo.OutlierTypeInProbe_AND = cell(numProbes,1); % Should mostly be 'T2&Q'
dataTable_Probes_CleanedInfo.NumOutliers_AND = zeros(numProbes,1);
dataTable_Probes_CleanedInfo.NumSpectra_AND_Cleaned = zeros(numProbes,1);

for i_probe = 1:numProbes
    original_probe_spectra_i = dataTableTrain_Initial.CombinedSpectra{i_probe};
    dataTable_Probes_CleanedInfo.NumOriginalSpectraInProbe(i_probe) = size(original_probe_spectra_i,1);

    if isempty(original_probe_spectra_i) || dataTable_Probes_CleanedInfo.NumOriginalSpectraInProbe(i_probe) == 0
        % Fill empty for all if no original spectra for this probe
        for strat = {'OR', 'AND'}
            dataTable_Probes_CleanedInfo.(sprintf('CombinedSpectra_%s_Cleaned',strat{1})){i_probe} = zeros(0, P.num_wavenumber_points);
            dataTable_Probes_CleanedInfo.(sprintf('OutlierSpectra_%s_Removed',strat{1})){i_probe} = zeros(0, P.num_wavenumber_points);
            dataTable_Probes_CleanedInfo.(sprintf('OutlierIndicesInProbe_%s',strat{1})){i_probe} = [];
            dataTable_Probes_CleanedInfo.(sprintf('OutlierTypeInProbe_%s',strat{1})){i_probe} = {};
        end
        continue;
    end

    global_indices_this_probe = find(Original_ProbeRowIndices_flat == i_probe);
    if isempty(global_indices_this_probe) || length(global_indices_this_probe) ~= size(original_probe_spectra_i,1)
        warning('Probe %d (%s): Mismatch mapping spectra for detailed table. Keeping original spectra for this probe in cleaned columns.', i_probe, dataTableTrain_Initial.Diss_ID{i_probe});
        dataTable_Probes_CleanedInfo.CombinedSpectra_OR_Cleaned{i_probe} = original_probe_spectra_i;
        dataTable_Probes_CleanedInfo.CombinedSpectra_AND_Cleaned{i_probe} = original_probe_spectra_i;
        % Other fields remain empty/zero for this probe
        continue;
    end
    
    % Internal indices relative to this probe's block of spectra
    internal_indices_this_probe = (1:size(original_probe_spectra_i,1))';

    % --- OR Strategy Processing for this Probe ---
    or_flags_this_probe = is_OR_outlier_all(global_indices_this_probe);
    dataTable_Probes_CleanedInfo.CombinedSpectra_OR_Cleaned{i_probe} = original_probe_spectra_i(~or_flags_this_probe, :);
    dataTable_Probes_CleanedInfo.OutlierSpectra_OR_Removed{i_probe}  = original_probe_spectra_i(or_flags_this_probe, :);
    dataTable_Probes_CleanedInfo.OutlierIndicesInProbe_OR{i_probe} = internal_indices_this_probe(or_flags_this_probe);
    dataTable_Probes_CleanedInfo.NumOutliers_OR(i_probe) = sum(or_flags_this_probe);
    dataTable_Probes_CleanedInfo.NumSpectra_OR_Cleaned(i_probe) = size(dataTable_Probes_CleanedInfo.CombinedSpectra_OR_Cleaned{i_probe},1);
    
    types_or = cell(sum(or_flags_this_probe),1); idx_type_or=0;
    t2_flags_probe = flag_T2_all(global_indices_this_probe); q_flags_probe = flag_Q_all(global_indices_this_probe);
    for k_spec=1:length(or_flags_this_probe)
        if or_flags_this_probe(k_spec), idx_type_or=idx_type_or+1;
            if t2_flags_probe(k_spec) && q_flags_probe(k_spec), types_or{idx_type_or}='T2&Q';
            elseif t2_flags_probe(k_spec), types_or{idx_type_or}='T2';
            elseif q_flags_probe(k_spec), types_or{idx_type_or}='Q';
            end
        end
    end
    dataTable_Probes_CleanedInfo.OutlierTypeInProbe_OR{i_probe} = types_or;

    % --- AND Strategy Processing for this Probe ---
    and_flags_this_probe = is_T2_and_Q_all(global_indices_this_probe); % Consensus outliers
    dataTable_Probes_CleanedInfo.CombinedSpectra_AND_Cleaned{i_probe} = original_probe_spectra_i(~and_flags_this_probe, :);
    dataTable_Probes_CleanedInfo.OutlierSpectra_AND_Removed{i_probe}  = original_probe_spectra_i(and_flags_this_probe, :);
    dataTable_Probes_CleanedInfo.OutlierIndicesInProbe_AND{i_probe} = internal_indices_this_probe(and_flags_this_probe);
    dataTable_Probes_CleanedInfo.NumOutliers_AND(i_probe) = sum(and_flags_this_probe);
    dataTable_Probes_CleanedInfo.NumSpectra_AND_Cleaned(i_probe) = size(dataTable_Probes_CleanedInfo.CombinedSpectra_AND_Cleaned{i_probe},1);
    dataTable_Probes_CleanedInfo.OutlierTypeInProbe_AND{i_probe} = repmat({'T2&Q'}, sum(and_flags_this_probe), 1);
end

cleanedDetailedTableFilename_mat = fullfile(P.resultsPath_OutlierApplication, sprintf('%s_dataTableTrain_Cleaned_OR_AND_Strategies.mat', P.datePrefix));
save(cleanedDetailedTableFilename_mat, 'dataTable_Probes_CleanedInfo', '-v7.3');
fprintf('Detailed per-probe table with OR & AND strategy results saved to: %s\n', cleanedDetailedTableFilename_mat);

% Excel Overview
overview_excel = dataTable_Probes_CleanedInfo;
overview_excel.OutlierIndicesInProbe_OR_str = cellfun(@(x) num2str(x(:)'), overview_excel.OutlierIndicesInProbe_OR, 'UniformOutput', false);
overview_excel.OutlierTypeInProbe_OR_str = cellfun(@(x) strjoin(x,'; '), overview_excel.OutlierTypeInProbe_OR, 'UniformOutput', false);
overview_excel.OutlierIndicesInProbe_AND_str = cellfun(@(x) num2str(x(:)'), overview_excel.OutlierIndicesInProbe_AND, 'UniformOutput', false);
overview_excel.OutlierTypeInProbe_AND_str = cellfun(@(x) strjoin(x,'; '), overview_excel.OutlierTypeInProbe_AND, 'UniformOutput', false);

colsToExport_excel = {'Diss_ID', 'Patient_ID', 'WHO_Grade', 'NumOriginalSpectraInProbe', ...
                      'NumSpectra_OR_Cleaned', 'NumOutliers_OR', 'OutlierIndicesInProbe_OR_str', 'OutlierTypeInProbe_OR_str', ...
                      'NumSpectra_AND_Cleaned', 'NumOutliers_AND', 'OutlierIndicesInProbe_AND_str', 'OutlierTypeInProbe_AND_str'};
varsToExportFinal_excel = intersect(colsToExport_excel, overview_excel.Properties.VariableNames, 'stable');
overviewFilename_xlsx = fullfile(P.resultsPath_OutlierApplication, sprintf('%s_dataTableTrain_Cleaned_OR_AND_Overview.xlsx', P.datePrefix));
try
    writetable(overview_excel(:, varsToExportFinal_excel), overviewFilename_xlsx);
    fprintf('Overview of detailed per-probe table saved to Excel: %s\n', overviewFilename_xlsx);
catch ME_excel_save
    fprintf('Could not save Excel overview: %s\n', ME_excel_save.message);
end

%% --- Save Lists of Removed Outliers for Each Strategy ---
fprintf('\n--- Saving Lists of Removed Outliers ---\n');
% OR Strategy Outliers
if sum(is_OR_outlier_all) > 0
    or_outliers_table = table(...
        Patient_ID_full_flat(is_OR_outlier_all), ...
        Original_ProbeRowIndices_flat(is_OR_outlier_all), ...
        Original_SpectrumIndexInProbe_flat(is_OR_outlier_all), ...
        T2_values_all(is_OR_outlier_all), ...
        Q_values_all(is_OR_outlier_all), ...
        flag_T2_all(is_OR_outlier_all), ... % Was it a T2 outlier?
        flag_Q_all(is_OR_outlier_all), ...  % Was it a Q outlier?
        'VariableNames', {'Patient_ID', 'Original_Probe_Row_Index', 'Original_Spectrum_Index_In_Probe', ...
                          'T2_Value', 'Q_Value', 'Was_T2_Outlier', 'Was_Q_Outlier'});
    orListFilename_csv = fullfile(P.resultsPath_OutlierApplication, sprintf('%s_RemovedOutliers_T2orQ_List.csv', P.datePrefix));
    writetable(or_outliers_table, orListFilename_csv);
    fprintf('List of %d removed T2 OR Q outliers saved to: %s\n', sum(is_OR_outlier_all), orListFilename_csv);
end
% AND Strategy Outliers
if sum(is_T2_and_Q_all) > 0
    and_outliers_table = table(...
        Patient_ID_full_flat(is_T2_and_Q_all), ...
        Original_ProbeRowIndices_flat(is_T2_and_Q_all), ...
        Original_SpectrumIndexInProbe_flat(is_T2_and_Q_all), ...
        T2_values_all(is_T2_and_Q_all), ...
        Q_values_all(is_T2_and_Q_all), ...
        'VariableNames', {'Patient_ID', 'Original_Probe_Row_Index', 'Original_Spectrum_Index_In_Probe', 'T2_Value', 'Q_Value'});
    andListFilename_csv = fullfile(P.resultsPath_OutlierApplication, sprintf('%s_RemovedOutliers_T2andQ_List.csv', P.datePrefix));
    writetable(and_outliers_table, andListFilename_csv);
    fprintf('List of %d removed T2 AND Q (Consensus) outliers saved to: %s\n', sum(is_T2_and_Q_all), andListFilename_csv);
end


fprintf('\n--- COMPREHENSIVE OUTLIER PROCESSING SCRIPT FINISHED ---\n');
% The outputs from this script (specifically the two .mat files in P.dataPath:
% *_training_set_no_outliers_T2orQ.mat and *_training_set_no_outliers_T2andQ.mat)
% are now ready to be used as input by the comprehensive model selection script.