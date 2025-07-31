%% run_pca_overview.m
% Perform a basic PCA on the training spectra and generate
% two scatter plots showing PC1 vs PC2 for WHO-1 and WHO-3
% samples separately.

clear; clc; close all;

P = setup_project_paths(pwd);
figuresPath = fullfile(P.figuresPath, 'Phase1_PCA');
if ~isfolder(figuresPath); mkdir(figuresPath); end

load(fullfile(P.dataPath,'data_table_train.mat'),'dataTableTrain');
load(fullfile(P.dataPath,'wavenumbers.mat'),'wavenumbers_roi');
if iscolumn(wavenumbers_roi); wavenumbers_roi = wavenumbers_roi'; end

[X, y_num] = flatten_spectra_for_pca(dataTableTrain, length(wavenumbers_roi));
[coeff, score, ~, ~, explained] = pca(X);

%% Plot WHO-1
fig1 = figure('Visible','off');
idx1 = y_num == 1;
scatter(score(idx1,1), score(idx1,2), 15, 'filled');
xlabel(sprintf('PC1 (%.1f%%)', explained(1))); 
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
title('PCA Scores - WHO-1');
exportgraphics(fig1, fullfile(figuresPath,'PCA_WHO1.png')); close(fig1);

%% Plot WHO-3
fig2 = figure('Visible','off');
idx3 = y_num == 3;
scatter(score(idx3,1), score(idx3,2), 15, 'filled');
xlabel(sprintf('PC1 (%.1f%%)', explained(1))); 
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
title('PCA Scores - WHO-3');
exportgraphics(fig2, fullfile(figuresPath,'PCA_WHO3.png')); close(fig2);

fprintf('PCA overview plots saved to %s\n', figuresPath);
