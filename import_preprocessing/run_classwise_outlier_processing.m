%% run_classwise_outlier_processing.m
% Perform PCA-based outlier detection separately for WHO-1 and WHO-3
% spectra. The script generates a cleaned training set with outliers
% removed and saves PCA results for each class.
%
% Date: 2025-11-27

clear; clc; close all;

P = setup_project_paths(pwd);
P.dataPath = P.dataPath;
P.resultsPath_OutlierExploration = fullfile(P.resultsPath, 'Phase1_OutlierExploration');
P.figuresPath_OutlierExploration = fullfile(P.figuresPath, 'Phase1_OutlierExploration');
if ~isfolder(P.resultsPath_OutlierExploration); mkdir(P.resultsPath_OutlierExploration); end
if ~isfolder(P.figuresPath_OutlierExploration); mkdir(P.figuresPath_OutlierExploration); end

alpha = 0.05;
varianceToExplain = 0.95;

trainFile = fullfile(P.dataPath, 'data_table_train.mat');
if ~isfile(trainFile)
    error('Training table not found: %s', trainFile);
end
load(trainFile, 'dataTableTrain');

wFile = fullfile(P.dataPath, 'wavenumbers.mat');
load(wFile, 'wavenumbers_roi');
if iscolumn(wavenumbers_roi); wavenumbers_roi = wavenumbers_roi'; end
numPts = length(wavenumbers_roi);

[X, y_num, y_cat, pid, probeIdx, specIdx] = flatten_spectra_for_pca(dataTableTrain, numPts);

classes = unique(y_cat);
resultsPerClass = struct();
allFlags = false(size(y_num));

for i = 1:numel(classes)
    classMask = y_cat == classes(i);
    res = compute_pca_t2_q(X(classMask,:), alpha, varianceToExplain);
    resultsPerClass(i).grade = char(classes(i));
    resultsPerClass(i).pca = res;
    allFlags(classMask) = res.is_outlier;

    % Generate visualizations for this WHO grade
    P_vis = P;
    gradeStr = char(classes(i));
    P_vis.figuresPath_OutlierExploration = fullfile(P.figuresPath_OutlierExploration, ['WHO_' gradeStr]);
    if ~isfolder(P_vis.figuresPath_OutlierExploration)
        mkdir(P_vis.figuresPath_OutlierExploration);
    end
    is_T2_and_Q = res.flag_T2 & res.flag_Q;
    visualize_outlier_exploration(X(classMask,:), ...
        y_num(classMask), y_cat(classMask), pid(classMask), ...
        wavenumbers_roi, ...
        res.score, res.explained, res.coeff, res.k_model, ...
        res.T2_values, res.Q_values, res.T2_threshold, res.Q_threshold, ...
        res.flag_T2, res.flag_Q, res.is_T2_only, res.is_Q_only, ...
        is_T2_and_Q, res.is_outlier, res.is_normal, ...
        P_vis);
end

cleanMask = ~allFlags;
X_clean = X(cleanMask,:);
y_clean_cat = y_cat(cleanMask);
y_clean_num = y_num(cleanMask);
PID_clean = pid(cleanMask);
probe_clean = probeIdx(cleanMask);
spec_clean = specIdx(cleanMask);

save(fullfile(P.dataPath,'training_set_no_outliers.mat'), ...
    'X_clean','y_clean_cat','y_clean_num','PID_clean','probe_clean','spec_clean','wavenumbers_roi');

save(fullfile(P.resultsPath,'classwise_outlier_detection.mat'), 'resultsPerClass','allFlags');

fprintf('Classwise outlier detection finished. Removed %d spectra.\n', sum(allFlags));
