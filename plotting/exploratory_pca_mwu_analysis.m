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
%   2. Extended PCA figure combining PC1 vs PC2, PC2 vs PC3, and a 3D
%      PC1/PC2/PC3 view coloured by WHO grade.
%   3. Mann--Whitney U p-value plot across wavenumbers with annotations
%      for the most significant differences.
%   4. Volcano plot contrasting effect size (median difference) against
%      statistical significance.
%   5. Mean spectra with \pm 1 standard deviation shading for each group
%      displayed in a dual-panel layout.
%   6. Spectral heatmaps for individual spectra (grouped by WHO grade)
%      and group-wise means.
%   7. Command-window summary of top wavenumbers ranked by significance.
%   8. Table of differentially absorbed wavenumbers (DAWNs) available in
%      the MATLAB workspace (and optional CSV export).
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
    whoGradesRaw = analyzeTable.WHO_Grade;
    keepMask = true(height(analyzeTable), 1);
    whoGradesText = {};

    if iscategorical(whoGradesRaw)
        whoGradesText = cellstr(whoGradesRaw);
    elseif isstring(whoGradesRaw)
        whoGradesText = cellstr(whoGradesRaw);
    elseif iscellstr(whoGradesRaw)
        whoGradesText = whoGradesRaw;
    elseif isnumeric(whoGradesRaw)
        warning(['WHO_Grade is numeric; expecting WHO-1/WHO-3 labels. ', ...
                 'Skipping grade-based filtering.']);
    else
        error('Unsupported data type for WHO_Grade column: %s.', class(whoGradesRaw));
    end

    if ~isempty(whoGradesText)
        keepMask = ismember(whoGradesText, {'WHO-1', 'WHO-3'});
    end

    analyzeTable = analyzeTable(keepMask, :);

    if height(analyzeTable) == 0
        error(['No WHO-1/WHO-3 entries remain after filtering. ', ...
               'Check the WHO_Grade values in the selected dataset.']);
    end
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

%% Extended PCA views (PC1 vs PC2, PC2 vs PC3, and 3D)
figure('Name', sprintf('Extended PCA Views: %s WHO-1 vs WHO-3', datasetLabel), ...
       'Position', [120 120 1000 800]);
tlPCA = tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% Tile 1: PC1 vs PC2 (replicates key view for comparison)
nexttile(tlPCA);
hold on; box on; grid on;
scatter(score(idxWHO1,1), score(idxWHO1,2), markerSize, 'o', ...
    'MarkerFaceColor', colorWHO1, 'MarkerEdgeColor', 'k', 'LineWidth', 0.75, ...
    'DisplayName', 'WHO-1');
scatter(score(idxWHO3,1), score(idxWHO3,2), markerSize, 's', ...
    'MarkerFaceColor', colorWHO3, 'MarkerEdgeColor', 'k', 'LineWidth', 0.75, ...
    'DisplayName', 'WHO-3');

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
title('PC1 vs PC2', 'Interpreter', 'latex');
set(gca, 'FontSize', 12, 'TickLabelInterpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
hold off;

% Tile 2: PC2 vs PC3
nexttile(tlPCA);
hold on; box on; grid on;
scatter(score(idxWHO1,2), score(idxWHO1,3), markerSize, 'o', ...
    'MarkerFaceColor', colorWHO1, 'MarkerEdgeColor', 'k', 'LineWidth', 0.75, ...
    'DisplayName', 'WHO-1');
scatter(score(idxWHO3,2), score(idxWHO3,3), markerSize, 's', ...
    'MarkerFaceColor', colorWHO3, 'MarkerEdgeColor', 'k', 'LineWidth', 0.75, ...
    'DisplayName', 'WHO-3');

scoreWHO1_23 = score(idxWHO1, 2:3);
scoreWHO3_23 = score(idxWHO3, 2:3);

if size(scoreWHO1_23,1) >= 3
    hull1_23 = convhull(scoreWHO1_23(:,1), scoreWHO1_23(:,2));
    patch(scoreWHO1_23(hull1_23,1), scoreWHO1_23(hull1_23,2), colorWHO1, ...
        'FaceAlpha', 0.18, 'EdgeColor', colorWHO1, 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');
end

if size(scoreWHO3_23,1) >= 3
    hull3_23 = convhull(scoreWHO3_23(:,1), scoreWHO3_23(:,2));
    patch(scoreWHO3_23(hull3_23,1), scoreWHO3_23(hull3_23,2), colorWHO3, ...
        'FaceAlpha', 0.18, 'EdgeColor', colorWHO3, 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');
end

xlabel(sprintf('PC2 (%.1f%%)', explained(2)), 'Interpreter', 'latex');
ylabel(sprintf('PC3 (%.1f%%)', explained(3)), 'Interpreter', 'latex');
title('PC2 vs PC3', 'Interpreter', 'latex');
set(gca, 'FontSize', 12, 'TickLabelInterpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
hold off;

% Tile 3: 3D scatter spanning both columns
nexttile(tlPCA, [1 2]);
hold on; box on; grid on;
scatter3(score(idxWHO1,1), score(idxWHO1,2), score(idxWHO1,3), markerSize, 'o', ...
    'MarkerFaceColor', colorWHO1, 'MarkerEdgeColor', 'k', 'LineWidth', 0.75, ...
    'DisplayName', 'WHO-1');
scatter3(score(idxWHO3,1), score(idxWHO3,2), score(idxWHO3,3), markerSize, 's', ...
    'MarkerFaceColor', colorWHO3, 'MarkerEdgeColor', 'k', 'LineWidth', 0.75, ...
    'DisplayName', 'WHO-3');
xlabel(sprintf('PC1 (%.1f%%)', explained(1)), 'Interpreter', 'latex');
ylabel(sprintf('PC2 (%.1f%%)', explained(2)), 'Interpreter', 'latex');
zlabel(sprintf('PC3 (%.1f%%)', explained(3)), 'Interpreter', 'latex');
title('3D PCA: PC1 vs PC2 vs PC3', 'Interpreter', 'latex');
set(gca, 'FontSize', 12, 'TickLabelInterpreter', 'latex');
view(45, 28);
legend('Location', 'best', 'Interpreter', 'latex');
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
%  6. Volcano plot (effect size vs significance)
% ----------------------------------------------------------------------
medianDiff = medianWHO3 - medianWHO1;
negLogP = -log10(p_values);
% Threshold (abs median difference) used to emphasise large effect sizes.
effectSizeThreshold = 0.05;

figure('Name', sprintf('Volcano Plot: %s', datasetLabel), ...
       'Position', [150 150 780 520]);
hold on; box on; grid on;
scatter(medianDiff, negLogP, 24, [0.3 0.3 0.3], 'filled', ...
    'DisplayName', 'All wavenumbers');

sigMask = p_values < 0.05;
effectMask = abs(medianDiff) > effectSizeThreshold;
highlightMask = sigMask & effectMask;

if any(highlightMask)
    scatter(medianDiff(highlightMask), negLogP(highlightMask), 40, [0.80 0.20 0.20], 'filled', ...
        'DisplayName', sprintf('p < 0.05 & |median diff| > %.3f', effectSizeThreshold));
end

xline(0, '--k', 'LineWidth', 1.1, 'DisplayName', '\Delta median = 0');
yline(-log10(0.05), '--r', 'LineWidth', 1.1, 'DisplayName', 'p = 0.05');

xlabel('Median Difference (WHO-3 $-$ WHO-1)', 'Interpreter', 'latex');
ylabel('$-\log_{10}(p\text{-value})$', 'Interpreter', 'latex');
title(sprintf('Volcano Plot of Spectral Differences (%s Set)', datasetLabel), 'Interpreter', 'latex');
set(gca, 'FontSize', 12, 'TickLabelInterpreter', 'latex');

[~, topVolcanoIdx] = sort(p_values);
topNVolcano = min(5, numel(topVolcanoIdx));
for vIdx = 1:topNVolcano
    idxNow = topVolcanoIdx(vIdx);
    text(medianDiff(idxNow), negLogP(idxNow) + 0.2, ...
        sprintf('%.0f cm$^{-1}$', wavenumbers_roi(idxNow)), ...
        'Interpreter', 'latex', 'HorizontalAlignment', 'center', 'FontSize', 10);
end

legend('Location', 'best', 'Interpreter', 'latex');
hold off;

%% ---------------------------------------------------------------------
%  7. Mean spectra with \pm 1 standard deviation shading
% ----------------------------------------------------------------------
meanWHO1 = mean(X_flat(idxWHO1, :), 1, 'omitnan');
meanWHO3 = mean(X_flat(idxWHO3, :), 1, 'omitnan');
stdWHO1  = std(X_flat(idxWHO1, :), 0, 1, 'omitnan');
stdWHO3  = std(X_flat(idxWHO3, :), 0, 1, 'omitnan');

figure('Name', sprintf('Mean Spectra: %s', datasetLabel), ...
       'Position', [140 140 960 520]);
tlMean = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% WHO-1 mean \pm SD
nexttile(tlMean);
hold on; box on; grid on;
fill([wavenumbers_roi, fliplr(wavenumbers_roi)], ...
     [meanWHO1 - stdWHO1, fliplr(meanWHO1 + stdWHO1)], ...
     colorWHO1, 'FaceAlpha', 0.20, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(wavenumbers_roi, meanWHO1, 'Color', colorWHO1, 'LineWidth', 2.0, ...
    'DisplayName', 'WHO-1 Mean');
set(gca, 'XDir', 'reverse', 'FontSize', 12, 'TickLabelInterpreter', 'latex');
xlabel('Wavenumber (cm^{-1})', 'Interpreter', 'latex');
ylabel('Absorbance (A.U.)', 'Interpreter', 'latex');
title('WHO-1 Mean Spectrum', 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
hold off;

% WHO-3 mean \pm SD
nexttile(tlMean);
hold on; box on; grid on;
fill([wavenumbers_roi, fliplr(wavenumbers_roi)], ...
     [meanWHO3 - stdWHO3, fliplr(meanWHO3 + stdWHO3)], ...
     colorWHO3, 'FaceAlpha', 0.20, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(wavenumbers_roi, meanWHO3, 'Color', colorWHO3, 'LineWidth', 2.0, ...
    'DisplayName', 'WHO-3 Mean');
set(gca, 'XDir', 'reverse', 'FontSize', 12, 'TickLabelInterpreter', 'latex');
xlabel('Wavenumber (cm^{-1})', 'Interpreter', 'latex');
ylabel('Absorbance (A.U.)', 'Interpreter', 'latex');
title('WHO-3 Mean Spectrum', 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
hold off;

sgtitle(tlMean, 'Mean FT-IR Spectra \pm 1 SD', 'Interpreter', 'latex');

%% ---------------------------------------------------------------------
%  8. Spectral heatmaps
% ----------------------------------------------------------------------

% Heatmap of all spectra grouped by WHO grade
figure('Name', sprintf('All Spectra Heatmap: %s', datasetLabel), ...
       'Position', [160 160 900 420]);
groupedSpectra = [X_flat(idxWHO1, :); X_flat(idxWHO3, :)];
imagesc(wavenumbers_roi, 1:size(groupedSpectra, 1), groupedSpectra);
colormap(parula);
colorbar;
set(gca, 'XDir', 'reverse', 'FontSize', 12, 'TickLabelInterpreter', 'latex');
xlabel('Wavenumber (cm$^{-1}$)', 'Interpreter', 'latex');
ylabel('Spectrum Index', 'Interpreter', 'latex');
title('All Spectra Heatmap (Grouped by WHO Grade)', 'Interpreter', 'latex');

% Draw separator line between WHO-1 and WHO-3 groups if both present
if any(idxWHO1) && any(idxWHO3)
    hold on;
    yline(sum(idxWHO1) + 0.5, '--k', 'LineWidth', 1.0, 'HandleVisibility', 'off');
    hold off;
end

% Mean spectra heatmap
figure('Name', sprintf('Mean Spectra Heatmap: %s', datasetLabel), ...
       'Position', [160 160 820 300]);
imagesc(wavenumbers_roi, 1:2, [meanWHO1; meanWHO3]);
colormap(parula);
colorbar;
set(gca, 'XDir', 'reverse', 'YTick', [1 2], 'YTickLabel', {'WHO-1', 'WHO-3'}, ...
    'FontSize', 12, 'TickLabelInterpreter', 'latex');
xlabel('Wavenumber (cm$^{-1}$)', 'Interpreter', 'latex');
ylabel('Group', 'Interpreter', 'latex');
title('Mean Spectra Heatmap', 'Interpreter', 'latex');

%% ---------------------------------------------------------------------
%  9. Print ranked summary of significant wavenumbers
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

%% ---------------------------------------------------------------------
% 10. Table of differentially absorbed wavenumbers (DAWNs)
% ----------------------------------------------------------------------
if ~isempty(sigIdx)
    dawnTable = table( ...
        wavenumbers_roi(sigIdx)', ...
        p_values(sigIdx)', ...
        medianWHO1(sigIdx)', ...
        medianWHO3(sigIdx)', ...
        medianDiff(sigIdx)', ...
        'VariableNames', {'Wavenumber_cm1', 'p_value', 'Median_WHO1', 'Median_WHO3', 'Median_Diff'});

    dawnTable = sortrows(dawnTable, 'p_value', 'ascend');

    assignin('base', 'dawnTable', dawnTable);

    % Optionally save table to disk by uncommenting the line below.
    % writetable(dawnTable, fullfile(P.resultsPath, sprintf('DAWN_table_%s.csv', lower(datasetLabel))));
else
    dawnTable = table([], [], [], [], [], ...
        'VariableNames', {'Wavenumber_cm1', 'p_value', 'Median_WHO1', 'Median_WHO3', 'Median_Diff'});
    assignin('base', 'dawnTable', dawnTable);
end

%% End of script
