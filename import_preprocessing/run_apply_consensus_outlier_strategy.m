% run_apply_consensus_outlier_strategy.m
%
% Applies a consensus (T2 AND Q) outlier removal strategy to an existing
% cleaned data table (dataTableTrain_cleaned, which was based on T2 OR Q).
% It adds new columns reflecting this consensus strategy.
%
% Prerequisites:
% 1. 'dataTableTrain.mat': Contains the original 'dataTableTrain' table 
%    (training probes BEFORE any outlier removal, with 'CombinedRawSpectra' or
%     'CombinedSpectra' holding the initially processed spectra for each probe).
% 2. '*_PCA_HotellingT2_Q_OutlierInfo.mat': Output from run_outlier_detection_pca2.m,
%    MODIFIED to contain 'T2_values_ALL_SPECTRA' and 'Q_values_ALL_SPECTRA'.
% 3. 'data_table_train_outliersremoved.mat': Contains 'dataTableTrain_cleaned'
%    (the table already processed by run_outlier_detection_pca2.m with T2 OR Q logic).
%    We will load this and add new columns to it.
%
% Date: 2025-05-17

%% 0. Initialization
clear; clc; close all;
fprintf('Applying Consensus Outlier Strategy - %s\n', string(datetime('now')));

P = setup_project_paths();

addpath(P.helperFunPath);

dataPath    = P.dataPath;
resultsPath = P.resultsPath; % Where outlierInfo and cleaned tables are
if ~exist(resultsPath, 'dir'), mkdir(resultsPath); end
dateStr = string(datetime('now','Format','yyyyMMdd'));

%% 1. Load Necessary Data
fprintf('Loading data...\n');

% --- Load original dataTableTrain (before any outlier removal by run_outlier_detection_pca2.m) ---
dataTableTrain_original_file = fullfile(dataPath, 'data_table_train.mat');
if ~exist(dataTableTrain_original_file, 'file')
    error('File "data_table_train.mat" not found in %s. This should be the training set table BEFORE outlier removal.', dataPath);
end
load(dataTableTrain_original_file, 'dataTableTrain'); % Assumes variable name is dataTableTrain
fprintf('Loaded original dataTableTrain (%d probes).\n', height(dataTableTrain));

% --- Load the OutlierInfo struct (which now must contain T2_values_ALL_SPECTRA and Q_values_ALL_SPECTRA) ---
outlierInfoFiles = dir(fullfile(resultsPath, '*_PCA_HotellingT2_Q_OutlierInfo.mat'));
if isempty(outlierInfoFiles)
    error('No *_PCA_HotellingT2_Q_OutlierInfo.mat file found in %s. Run modified run_outlier_detection_pca2.m first.', resultsPath);
end
[~,idxSort] = sort([outlierInfoFiles.datenum],'descend');
latestOutlierInfoFile = fullfile(resultsPath, outlierInfoFiles(idxSort(1)).name);
fprintf('Loading outlier info from: %s\n', latestOutlierInfoFile);
loadedOutlierInfo = load(latestOutlierInfoFile, 'outlierInfo');
if ~isfield(loadedOutlierInfo, 'outlierInfo') || ...
   ~isfield(loadedOutlierInfo.outlierInfo, 'T2_values_ALL_SPECTRA') || ...
   ~isfield(loadedOutlierInfo.outlierInfo, 'Q_values_ALL_SPECTRA')
    error('Loaded outlierInfo struct does not contain T2_values_ALL_SPECTRA or Q_values_ALL_SPECTRA. Please re-run modified run_outlier_detection_pca2.m.');
end
outlierInfo = loadedOutlierInfo.outlierInfo;
T2_values_all = outlierInfo.T2_values_ALL_SPECTRA;
Q_values_all  = outlierInfo.Q_values_ALL_SPECTRA;
T2_thresh     = outlierInfo.T2_threshold;
Q_thresh      = outlierInfo.Q_threshold;

% --- Load the existing dataTableTrain_cleaned (from data_table_train_outliersremoved.mat) ---
% This table was cleaned using T2 OR Q. We will add new columns to it.
cleanedTableFile = fullfile(resultsPath, 'data_table_train_outliersremoved.mat'); % Your specified file
if ~exist(cleanedTableFile, 'file')
    error('File "data_table_train_outliersremoved.mat" not found in %s.', resultsPath);
end
load(cleanedTableFile, 'dataTableTrain_cleaned'); % Assumes variable is dataTableTrain_cleaned
fprintf('Loaded dataTableTrain_cleaned (%d probes) from %s.\n', height(dataTableTrain_cleaned), cleanedTableFile);

% Reconstruct mapping from flat X_train (used in outlier detection) back to original table rows and intra-sample indices
% This requires Patient_ID_train and Original_Indices_train etc. which were used to build X_train in run_outlier_detection_pca2.m
% For this script, we need to reconstruct this mapping.
% Let's assume the original run_outlier_detection_pca2.m produced X_train and its mapping arrays.
% If these (Original_Indices_train, Original_Spectrum_Index_In_Sample_train) were not saved with outlierInfo,
% we need to rebuild them from dataTableTrain (the input to outlier detection).

temp_Original_Indices_train = [];
temp_Original_Spectrum_Index_In_Sample_train = [];
numSpectraTotal_check = 0;
for i = 1:height(dataTableTrain) % Iterate through original dataTableTrain
    numIndivSpec = size(dataTableTrain.CombinedRawSpectra{i}, 1); % Or CombinedSpectra if that was used for X_train
    if numIndivSpec > 0
        temp_Original_Indices_train = [temp_Original_Indices_train; repmat(i, numIndivSpec, 1)];
        temp_Original_Spectrum_Index_In_Sample_train = [temp_Original_Spectrum_Index_In_Sample_train; (1:numIndivSpec)'];
        numSpectraTotal_check = numSpectraTotal_check + numIndivSpec;
    end
end

if numSpectraTotal_check ~= length(T2_values_all)
    error('Mismatch between number of spectra derived from dataTableTrain (%d) and length of T2_values_all (%d). Ensure dataTableTrain used here is the same one used for outlier detection.', numSpectraTotal_check, length(T2_values_all));
end
Original_Indices_train_map = temp_Original_Indices_train;
Original_Spectrum_Index_In_Sample_train_map = temp_Original_Spectrum_Index_In_Sample_train;


%% 2. Determine Consensus Outliers (T2 AND Q)
% =========================================================================
fprintf('Determining consensus outliers (T2 AND Q)...\n');
is_T2_outlier_all = (T2_values_all > T2_thresh);
is_Q_outlier_all  = (Q_values_all > Q_thresh);
is_consensus_outlier_all = is_T2_outlier_all & is_Q_outlier_all; % Logical AND

num_consensus_outliers_total = sum(is_consensus_outlier_all);
fprintf('%d total consensus outlier spectra identified across all training probes.\n', num_consensus_outliers_total);

%% 3. Add New Consensus-Cleaned Columns to dataTableTrain_cleaned
% =========================================================================
fprintf('Adding consensus-based cleaned spectra and outlier info to dataTableTrain_cleaned...\n');

numProbesInCleanedTable = height(dataTableTrain_cleaned);
% Initialize new columns
dataTableTrain_cleaned.NumSpectra_consensus = zeros(numProbesInCleanedTable, 1);
dataTableTrain_cleaned.CombinedSpectra_consensus = cell(numProbesInCleanedTable, 1);
dataTableTrain_cleaned.OutlierSpectra_consensus = cell(numProbesInCleanedTable, 1);
dataTableTrain_cleaned.OutlierIndicesInSample_consensus = cell(numProbesInCleanedTable, 1);
dataTableTrain_cleaned.OutlierTypeInSample_consensus = cell(numProbesInCleanedTable,1); % T2, Q, T2&Q for consensus outliers
dataTableTrain_cleaned.NumOutliers_consensus = zeros(numProbesInCleanedTable, 1);

% We need to iterate through the original dataTableTrain to get original spectra
% and map the consensus outlier flags.
for i_probe = 1:numProbesInCleanedTable % Iterate through rows of dataTableTrain_cleaned
    % Find the corresponding row in the original dataTableTrain
    % This assumes Diss_ID is a unique key and present in both.
    current_Diss_ID_cleaned = dataTableTrain_cleaned.Diss_ID{i_probe};
    original_dtTrain_row_idx = find(strcmp(dataTableTrain.Diss_ID, current_Diss_ID_cleaned));
    
    if isempty(original_dtTrain_row_idx) || length(original_dtTrain_row_idx) > 1
        warning('Could not uniquely match Diss_ID "%s" from dataTableTrain_cleaned to original dataTableTrain. Skipping consensus for this probe.', current_Diss_ID_cleaned);
        % Fill with empty or NaN for this probe's consensus columns
        numOriginalSpectraInProbe = size(dataTableTrain.CombinedRawSpectra{original_dtTrain_row_idx(1)}, 1);
        dataTableTrain_cleaned.CombinedSpectra_consensus{i_probe} = zeros(0, num_wavenumber_points); 
        dataTableTrain_cleaned.OutlierSpectra_consensus{i_probe} = zeros(0, num_wavenumber_points);
        continue;
    end
    
    % Get the original raw spectra for this probe (before any outlier removal)
    % Assuming 'CombinedRawSpectra' holds the spectra that T2/Q were calculated on (after SG, SNV, L2)
    % If T2/Q were calculated on 'CombinedSpectra' (already processed), use that.
    % For this to align with X_train from run_outlier_detection_pca2.m, CombinedRawSpectra from dataTableTrain should be the input.
    original_spectra_this_probe = dataTableTrain.CombinedRawSpectra{original_dtTrain_row_idx};
    if isempty(original_spectra_this_probe)
        dataTableTrain_cleaned.CombinedSpectra_consensus{i_probe} = zeros(0, num_wavenumber_points);
        dataTableTrain_cleaned.OutlierSpectra_consensus{i_probe} = zeros(0, num_wavenumber_points);
        continue;
    end
    numOriginalSpectraInProbe = size(original_spectra_this_probe, 1);

    % Find global indices in T2_values_all that correspond to this probe's spectra
    global_indices_for_this_probe = find(Original_Indices_train_map == original_dtTrain_row_idx);
    
    if length(global_indices_for_this_probe) ~= numOriginalSpectraInProbe
        warning('Mismatch in spectra count for probe %s. Expected %d from T2/Q map, got %d from original table. Skipping consensus.', ...
            current_Diss_ID_cleaned, length(global_indices_for_this_probe), numOriginalSpectraInProbe);
        dataTableTrain_cleaned.CombinedSpectra_consensus{i_probe} = zeros(0, num_wavenumber_points);
        dataTableTrain_cleaned.OutlierSpectra_consensus{i_probe} = zeros(0, num_wavenumber_points);
        continue;
    end
    
    % Get consensus outlier status for this probe's spectra
    consensus_outlier_status_this_probe = is_consensus_outlier_all(global_indices_for_this_probe);
    
    % Get T2 only and Q only status for these consensus outliers for OutlierType
    T2_status_this_probe_outliers = is_T2_outlier_all(global_indices_for_this_probe(consensus_outlier_status_this_probe));
    Q_status_this_probe_outliers = is_Q_outlier_all(global_indices_for_this_probe(consensus_outlier_status_this_probe));

    
    dataTableTrain_cleaned.CombinedSpectra_consensus{i_probe} = original_spectra_this_probe(~consensus_outlier_status_this_probe, :);
    dataTableTrain_cleaned.OutlierSpectra_consensus{i_probe}  = original_spectra_this_probe(consensus_outlier_status_this_probe, :);
    
    % Indices are relative to the original block of spectra for that probe
    internal_indices_original = (1:numOriginalSpectraInProbe)';
    dataTableTrain_cleaned.OutlierIndicesInSample_consensus{i_probe} = internal_indices_original(consensus_outlier_status_this_probe);
    
    types_consensus_outliers = cell(sum(consensus_outlier_status_this_probe),1);
    idx_co = 0;
    for k_co = 1:length(consensus_outlier_status_this_probe) % Iterate through spectra of this probe
        if consensus_outlier_status_this_probe(k_co) % If it's a consensus outlier
            idx_co = idx_co + 1;
            % Since it's a consensus outlier, it's both T2 and Q.
            types_consensus_outliers{idx_co} = 'T2&Q'; 
        end
    end
    dataTableTrain_cleaned.OutlierTypeInSample_consensus{i_probe} = types_consensus_outliers;

    dataTableTrain_cleaned.NumSpectra_consensus(i_probe) = size(dataTableTrain_cleaned.CombinedSpectra_consensus{i_probe}, 1);
    dataTableTrain_cleaned.NumOutliers_consensus(i_probe) = sum(consensus_outlier_status_this_probe);
end

fprintf('New consensus-based columns added to dataTableTrain_cleaned.\n');
disp('Head of dataTableTrain_cleaned with new consensus columns (selected):');
disp(head(dataTableTrain_cleaned(:, ...
    {'Diss_ID', 'NumTotalSpectra', 'NumSpectra_consensus', 'NumOutliers_consensus', 'CombinedSpectra_consensus', 'OutlierIndicesInSample_consensus', 'OutlierTypeInSample_consensus'})));

%% 4. Save the Updated Table
% =========================================================================
% Save with a new name to distinguish from the original T2|Q cleaning
updatedCleanedTableFile = fullfile(resultsPath, 'data_table_train_cleaned_with_consensus_outliers.mat');
save(updatedCleanedTableFile, 'dataTableTrain_cleaned');
fprintf('Updated table with consensus outlier info saved to: %s\n', updatedCleanedTableFile);

% Optionally, save an Excel overview as well
overviewFilename_xlsx = fullfile(resultsPath, sprintf('%s_dataTableTrain_Overview_WithConsensus.xlsx', dateStr));
try
    dataTableTrain_cleaned_for_excel = dataTableTrain_cleaned;
    % Add NumOriginalSpectra if not already present (it should be in dataTableTrain_cleaned from run_outlier_detection_pca2.m)
    if ~ismember('NumOriginalSpectra', dataTableTrain_cleaned_for_excel.Properties.VariableNames)
         dataTableTrain_cleaned_for_excel.NumOriginalSpectra = cellfun(@(x) size(x,1), dataTableTrain.CombinedRawSpectra(ismember(dataTableTrain.Diss_ID, dataTableTrain_cleaned.Diss_ID)));
    end
    
    % The script run_outlier_detection_pca2.m already added NumCleanedSpectra (T2|Q) and NumOutlierSpectra (T2|Q)
    % We just added NumSpectra_consensus and NumOutliers_consensus
    
    dataTableTrain_cleaned_for_excel.OutlierIndicesInSample_consensus_str = cellfun(@(x) num2str(x(:)'), dataTableTrain_cleaned_for_excel.OutlierIndicesInSample_consensus, 'UniformOutput', false);
    dataTableTrain_cleaned_for_excel.OutlierTypeInSample_consensus_str = cellfun(@(x) strjoin(x,', '), dataTableTrain_cleaned_for_excel.OutlierTypeInSample_consensus, 'UniformOutput', false);

    colsToExport = {...
        'Diss_ID', 'Patient_ID', 'WHO_Grade', 'NumOriginalSpectra', ...
        'NumCleanedSpectra', 'NumOutlierSpectra', 'OutlierIndicesInSample_str', 'OutlierTypeInSample_str', ... % Original T2|Q
        'NumSpectra_consensus', 'NumOutliers_consensus', 'OutlierIndicesInSample_consensus_str', 'OutlierTypeInSample_consensus_str'}; % New Consensus
    
    varsToExportFinal = intersect(colsToExport, dataTableTrain_cleaned_for_excel.Properties.VariableNames, 'stable');
    
    writetable(dataTableTrain_cleaned_for_excel(:, varsToExportFinal), overviewFilename_xlsx);
    fprintf('Overview of table with consensus outlier info saved to Excel: %s\n', overviewFilename_xlsx);
catch ME_excel_save
    fprintf('Could not save Excel overview: %s\n', ME_excel_save.message);
end


fprintf('\nConsensus Outlier Strategy Application Complete.\n');