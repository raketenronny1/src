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
if ~contains(path, P.helperFunPath)
    addpath(P.helperFunPath);
end

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
[X_train_full_flat, y_numeric_full_flat, y_cat_full_flat, Patient_ID_full_flat, ...
    Original_ProbeRowIndices_flat, Original_SpectrumIndexInProbe_flat] = ...
    flatten_spectra_for_pca(dataTableTrain_Initial, P.num_wavenumber_points);
fprintf('Data for PCA prepared: %d spectra, %d features.\n', size(X_train_full_flat,1), size(X_train_full_flat,2));

%% 3. Perform PCA, Calculate T²/Q Statistics & Thresholds (ONCE)
fprintf('\n--- 3. PCA, T2/Q Calculation & Thresholds ---\n');

results_pca = compute_pca_t2_q(X_train_full_flat, P.alpha_T2_Q, P.variance_to_explain_for_PCA_model);

coeff_pca    = results_pca.coeff;
score_pca    = results_pca.score;
latent_pca   = results_pca.latent;
explained_pca= results_pca.explained;
mu_pca       = results_pca.mu;
k_model_pca  = results_pca.k_model;
T2_values_all= results_pca.T2_values;
Q_values_all = results_pca.Q_values;
T2_threshold = results_pca.T2_threshold;
Q_threshold  = results_pca.Q_threshold;
flag_T2_all  = results_pca.flag_T2;
flag_Q_all   = results_pca.flag_Q;
is_T2_only_all  = results_pca.is_T2_only;
is_Q_only_all   = results_pca.is_Q_only;
is_T2_and_Q_all = results_pca.is_T2_and_Q;
is_OR_outlier_all = results_pca.is_OR_outlier;
is_normal_all   = results_pca.is_normal;

fprintf('PCA completed. k_model=%d, T2 threshold=%.4f, Q threshold=%.4g\n', k_model_pca, T2_threshold, Q_threshold);
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