%% exploratory_pca_mwu_analysis.m
%
% Exploratory principal component analysis (PCA) and Mann--Whitney U
% (Wilcoxon rank-sum) testing for FT-IR spectra comparing WHO-1 vs WHO-3
% probes. This script aggregates spectra from the selected dataset,
% performs PCA for visualization, evaluates per-wavenumber statistical
% significance, and visualizes mean spectra with variability bands.
%
% Usage:
%   - Adjust the `analyzeDataset` variable below to switch between the
%     training and testing spectral tables ("train" or "test").
%   - The script assumes `setup_project_paths` and
%     `flatten_spectra_for_pca` are available on the MATLAB path, and that
%     the data tables only contain WHO-1 and WHO-3 probes (WHO-2 removed
%     beforehand).
%
% The output includes:
%   1. PCA scatter plot (PC1 vs PC2) with convex hulls per group.
%   2. Mann--Whitney U p-value plot across wavenumbers with annotations
%      for the most significant differences.
%   3. Mean spectra with \pm 1 standard deviation shading for each group.
%   4. Command-window summary of top wavenumbers ranked by significance.
%
% Date: 2025-05-20
%

%% ---------------------------------------------------------------------
%  0. User configuration
% ----------------------------------------------------------------------
% Choose which dataset to analyse: 'train' or 'test'.
analyzeDataset = 'train';

% Number of top-ranked wavenumbers to display in the textual summary.
topNToDisplay = 10;

%% ---------------------------------------------------------------------
%  1. Load data and flatten spectra
% ----------------------------------------------------------------------
P = setup_project_paths(pwd);
dataPath = P.dataPath;

trainPath = fullfile(dataPath, 'data_table_train.mat');
testPath  = fullfile(dataPath, 'data_table_test.mat');
wavenumberPath = fullfile(dataPath, 'wavenumbers.mat');

if exist(trainPath, 'file') ~= 2 || exist(testPath, 'file') ~= 2
    error('Required data tables were not found in %s.', dataPath);
end

load(trainPath, 'dataTableTrain');
load(testPath,  'dataTableTest');
load(wavenumberPath, 'wavenumbers_roi');

if iscolumn(wavenumbers_roi)
    wavenumbers_roi = wavenumbers_roi';
end

switch lower(string(analyzeDataset))
    case "train"
        analyzeTable = dataTableTrain;
        datasetLabel = 'Training';
    case "test"
        analyzeTable = dataTableTest;
        datasetLabel = 'Testing';
    otherwise
        error('Unknown analyzeDataset value: %s (use "train" or "test").', analyzeDataset);
end

% Filter to WHO-1 / WHO-3 rows only (in case WHO-2 slipped through).
if ismember('WHO_Grade', analyzeTable.Properties.VariableNames)
    analyzeTable = analyzeTable(ismember(string(analyzeTable.WHO_Grade), {"WHO-1","WHO-3"}), :);
end

[X_flat, y_num] = flatten_spectra_for_pca(analyzeTable, length(wavenumbers_roi));

if isempty(X_flat)
    error('No spectra were flattened. Check the input table contents.');
end

%% ---------------------------------------------------------------------
%  2. Perform PCA
% ----------------------------------------------------------------------
[~, score, ~, ~, explained] = pca(X_flat);

% Extract group indices for later use.
idxWHO1 = (y_num == 1);
idxWHO3 = (y_num == 3);

if ~any(idxWHO1) || ~any(idxWHO3)
    error('Both WHO-1 and WHO-3 groups must be present for analysis.');
end

%% ---------------------------------------------------------------------
%  3. PCA scatter plot with convex hulls
% ----------------------------------------------------------------------
figure('Name', sprintf('PCA Scores: %s WHO-1 vs WHO-3', datasetLabel), ...
       'Position', [100 100 720 600]);
hold on; box on; grid on;

colorWHO1 = [0.90, 0.60, 0.40];
colorWHO3 = [0.40, 0.70, 0.90];
markerSize = 42;

scatter(score(idxWHO1,1), score(idxWHO1,2), markerSize, 'o', ...
    'MarkerFaceColor', colorWHO1, 'MarkerEdgeColor', 'k', 'LineWidth', 0.75, ...
    'DisplayName', 'WHO-1');
scatter(score(idxWHO3,1), score(idxWHO3,2), markerSize, 's', ...
    'MarkerFaceColor', colorWHO3, 'MarkerEdgeColor', 'k', 'LineWidth', 0.75, ...
    'DisplayName', 'WHO-3');

scoreWHO1 = score(idxWHO1, 1:2);
scoreWHO3 = score(idxWHO3, 1:2);

if size(scoreWHO1,1) >= 3
    hull1 = convhull(scoreWHO1(:,1), scoreWHO1(:,2));
    patch(scoreWHO1(hull1,1), scoreWHO1(hull1,2), colorWHO1, ...
        'FaceAlpha', 0.18, 'EdgeColor', colorWHO1, 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');
end

if size(scoreWHO3,1) >= 3
    hull3 = convhull(scoreWHO3(:,1), scoreWHO3(:,2));
    patch(scoreWHO3(hull3,1), scoreWHO3(hull3,2), colorWHO3, ...
        'FaceAlpha', 0.18, 'EdgeColor', colorWHO3, 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');
end

xlabel(sprintf('PC1 (%.1f%%)', explained(1)), 'Interpreter', 'latex');
ylabel(sprintf('PC2 (%.1f%%)', explained(2)), 'Interpreter', 'latex');
title(sprintf('PCA: %s Set (WHO-1 vs WHO-3)', datasetLabel), 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
set(gca, 'FontSize', 12, 'TickLabelInterpreter', 'latex');
axis tight;
hold off;

%% ---------------------------------------------------------------------
%  4. Mann--Whitney U test per wavenumber
% ----------------------------------------------------------------------
numWavenumbers = length(wavenumbers_roi);
p_values = nan(1, numWavenumbers);
medianWHO1 = nan(1, numWavenumbers);
medianWHO3 = nan(1, numWavenumbers);

for wIdx = 1:numWavenumbers
    valsWHO1 = X_flat(idxWHO1, wIdx);
    valsWHO3 = X_flat(idxWHO3, wIdx);

    if all(valsWHO1 == valsWHO1(1)) && all(valsWHO3 == valsWHO3(1)) && valsWHO1(1) == valsWHO3(1)
        % Identical distributions -> p-value of 1.
        p_values(wIdx) = 1;
    else
        p_values(wIdx) = ranksum(valsWHO1, valsWHO3);
    end

    medianWHO1(wIdx) = median(valsWHO1, 'omitnan');
    medianWHO3(wIdx) = median(valsWHO3, 'omitnan');
end

% Avoid zero p-values for logarithmic plotting.
p_values(p_values == 0) = realmin;

MWU_results = struct( ...
    'wavenumber', num2cell(wavenumbers_roi), ...
    'p_value',    num2cell(p_values), ...
    'median_WHO1', num2cell(medianWHO1), ...
    'median_WHO3', num2cell(medianWHO3));

%% ---------------------------------------------------------------------
%  5. Plot p-values across wavenumbers
% ----------------------------------------------------------------------
figure('Name', sprintf('Mann-Whitney U p-values: %s', datasetLabel), ...
       'Position', [120 120 740 520]);
hold on; box on;

semilogy(wavenumbers_roi, p_values, 'Color', [0.10 0.30 0.60], 'LineWidth', 1.6, ...
    'DisplayName', 'p-value');

sigLine = yline(0.05, '--r', 'LineWidth', 1.2, 'DisplayName', 'p = 0.05');
if isprop(sigLine, 'Label')
    sigLine.Label = 'p = 0.05';
    sigLine.LabelHorizontalAlignment = 'left';
    sigLine.Interpreter = 'latex';
end

xlabel('Wavenumber (cm^{-1})', 'Interpreter', 'latex');
ylabel('p-value (Mann--Whitney U)', 'Interpreter', 'latex');
title(sprintf('Per-Wavenumber Significance: %s Set', datasetLabel), 'Interpreter', 'latex');
set(gca, 'XDir', 'reverse', 'YScale', 'log', 'FontSize', 12, ...
    'TickLabelInterpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');

affordableIdx = find(~isnan(p_values));
[~, sortedIdxLocal] = sort(p_values(affordableIdx), 'ascend');
numToAnnotate = min(5, numel(sortedIdxLocal));

for iIdx = 1:numToAnnotate
    actualIdx = affordableIdx(sortedIdxLocal(iIdx));
    if p_values(actualIdx) < 0.05
        plot(wavenumbers_roi(actualIdx), p_values(actualIdx), 'ko', ...
            'MarkerFaceColor', [0.85 0.10 0.10], 'MarkerSize', 6, ...
            'HandleVisibility', 'off');
        text(wavenumbers_roi(actualIdx), p_values(actualIdx) * 1.3, ...
            sprintf('%.0f cm^{-1}', wavenumbers_roi(actualIdx)), ...
            'Color', 'k', 'Interpreter', 'latex', 'FontSize', 10, ...
            'HorizontalAlignment', 'center');
    end
end

hold off;

%% ---------------------------------------------------------------------
%  6. Mean spectra with \pm 1 standard deviation shading
% ----------------------------------------------------------------------
meanWHO1 = mean(X_flat(idxWHO1, :), 1, 'omitnan');
meanWHO3 = mean(X_flat(idxWHO3, :), 1, 'omitnan');
stdWHO1  = std(X_flat(idxWHO1, :), 0, 1, 'omitnan');
stdWHO3  = std(X_flat(idxWHO3, :), 0, 1, 'omitnan');

figure('Name', sprintf('Mean Spectra: %s', datasetLabel), ...
       'Position', [140 140 760 520]);
hold on; box on; grid on;

plot(wavenumbers_roi, meanWHO1, 'Color', colorWHO1, 'LineWidth', 2.0, ...
    'DisplayName', 'WHO-1 Mean');
plot(wavenumbers_roi, meanWHO3, 'Color', colorWHO3, 'LineWidth', 2.0, ...
    'DisplayName', 'WHO-3 Mean');

fill([wavenumbers_roi, fliplr(wavenumbers_roi)], ...
     [meanWHO1 - stdWHO1, fliplr(meanWHO1 + stdWHO1)], ...
     colorWHO1, 'FaceAlpha', 0.20, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill([wavenumbers_roi, fliplr(wavenumbers_roi)], ...
     [meanWHO3 - stdWHO3, fliplr(meanWHO3 + stdWHO3)], ...
     colorWHO3, 'FaceAlpha', 0.20, 'EdgeColor', 'none', 'HandleVisibility', 'off');

set(gca, 'XDir', 'reverse', 'FontSize', 12, 'TickLabelInterpreter', 'latex');
xlabel('Wavenumber (cm^{-1})', 'Interpreter', 'latex');
ylabel('Absorbance (A.U.)', 'Interpreter', 'latex');
title(sprintf('Mean FT-IR Spectra \pm 1 SD (%s Set)', datasetLabel), 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
axis tight;
hold off;

%% ---------------------------------------------------------------------
%  7. Print ranked summary of significant wavenumbers
% ----------------------------------------------------------------------
validPIdx = find(~isnan(p_values));
[sortedP, sortedLocalIdx] = sort(p_values(validPIdx));

fprintf('\nTop %d most significant wavenumbers (dataset: %s):\n', ...
    min(topNToDisplay, numel(sortedP)), datasetLabel);
fprintf(' Rank | Wavenumber (cm^-1) |    p-value    | Median WHO-1 | Median WHO-3\n');
fprintf('---------------------------------------------------------------\n');
for rIdx = 1:min(topNToDisplay, numel(sortedP))
    globalIdx = validPIdx(sortedLocalIdx(rIdx));
    fprintf(' %3d  | %9.1f          | %10.3e | %11.4f | %11.4f\n', ...
        rIdx, wavenumbers_roi(globalIdx), p_values(globalIdx), ...
        medianWHO1(globalIdx), medianWHO3(globalIdx));
end

sigIdx = find(p_values < 0.05 & ~isnan(p_values));
fprintf('\nNumber of wavenumbers with p < 0.05: %d (out of %d).\n', ...
    numel(sigIdx), numWavenumbers);

%% End of script
