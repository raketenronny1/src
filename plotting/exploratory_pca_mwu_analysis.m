%% exploratory_pca_mwu_analysis.m
% Exploratory PCA and Mann-Whitney U Testing of FT-IR Spectra (WHO-1 vs WHO-3)

%% User Settings
analyzeDataset = 'train';
binSize = 4;
topNToDisplay = 10;

%% Load Data
P = setup_project_paths(pwd);
dataPath = P.dataPath;
load(fullfile(dataPath, 'data_table_train.mat'), 'dataTableTrain');
load(fullfile(dataPath, 'data_table_test.mat'),  'dataTableTest');
load(fullfile(dataPath, 'wavenumbers.mat'), 'wavenumbers_roi');

if iscolumn(wavenumbers_roi)
    wavenumbers_roi = wavenumbers_roi';
end

switch lower(analyzeDataset)
    case 'train', analyzeTable = dataTableTrain; datasetLabel = 'Training';
    case 'test',  analyzeTable = dataTableTest;  datasetLabel = 'Testing';
    otherwise, error('Invalid dataset.');
end

% Filter for WHO-1 / WHO-3
if ismember('WHO_Grade', analyzeTable.Properties.VariableNames)
    gradeStr = upper(strrep(string(analyzeTable.WHO_Grade), ' ', ''));
    keepMask = ismember(gradeStr, {'WHO-1','WHO3','WHO-3'});
    analyzeTable = analyzeTable(keepMask,:);
end

%% Flatten by MeanSpectrum (1 row per probe)
nWaves = length(wavenumbers_roi);
numProbes = height(analyzeTable);
X_flat_mean = nan(numProbes, nWaves);
y_num = nan(numProbes,1);

for i = 1:numProbes
    if ~ismissing(analyzeTable.WHO_Grade(i)) && ~isempty(analyzeTable.MeanSpectrum{i})
        ms = analyzeTable.MeanSpectrum{i};
        if size(ms,2) == nWaves
            X_flat_mean(i,:) = ms;
        end
        label = upper(strrep(string(analyzeTable.WHO_Grade(i)), ' ', ''));
        y_num(i) = strcmp(label, 'WHO-3') * 3 + strcmp(label, 'WHO-1') * 1;
    end
end

% Remove rows with all NaNs
validRows = all(~isnan(X_flat_mean), 2);
X_flat_mean = X_flat_mean(validRows, :);
y_num = y_num(validRows);

idxWHO1 = y_num == 1;
idxWHO3 = y_num == 3;

%% Limit ROI to 1800–950 cm⁻¹
roiMask = (wavenumbers_roi <= 1800) & (wavenumbers_roi >= 950);
wavenumbers_roi = wavenumbers_roi(roiMask);
X_flat_mean = X_flat_mean(:, roiMask);

%% PCA (on z-scored mean spectra in ROI)
[~,score,~,~,explained,coeff] = pca(zscore(X_flat_mean));
colors = {[0.90 0.60 0.40], [0.40 0.70 0.90]};

figure('Name', 'PCA Views (mean spectra)', 'Position', [100 100 1000 800]);
tiledlayout(2,2);

% PC1 vs PC2
nexttile;
scatter(score(idxWHO1,1),score(idxWHO1,2),36,colors{1},'filled'); hold on;
scatter(score(idxWHO3,1),score(idxWHO3,2),36,colors{2},'filled');
xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
title('PC1 vs PC2'); grid on;

% PC2 vs PC3
nexttile;
scatter(score(idxWHO1,2),score(idxWHO1,3),36,colors{1},'filled'); hold on;
scatter(score(idxWHO3,2),score(idxWHO3,3),36,colors{2},'filled');
xlabel(sprintf('PC2 (%.1f%%)', explained(2)));
ylabel(sprintf('PC3 (%.1f%%)', explained(3)));
title('PC2 vs PC3'); grid on;

% 3D PCA
nexttile([1 2]);
scatter3(score(idxWHO1,1),score(idxWHO1,2),score(idxWHO1,3),36,colors{1},'filled'); hold on;
scatter3(score(idxWHO3,1),score(idxWHO3,2),score(idxWHO3,3),36,colors{2},'filled');
xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
zlabel(sprintf('PC3 (%.1f%%)', explained(3)));
title('3D PCA'); grid on; view(45,30);


%% Binning before MWU
nBins = floor(length(wavenumbers_roi) / binSize);
p_vals = nan(1,nBins); med1 = p_vals; med3 = p_vals;
waveBinCenters = nan(1,nBins);

for b = 1:nBins
    idx = (b-1)*binSize+1 : b*binSize;
    vals1 = median(X_flat_mean(idxWHO1, idx),2);
    vals3 = median(X_flat_mean(idxWHO3, idx),2);

    vals1 = vals1(~isnan(vals1));
    vals3 = vals3(~isnan(vals3));

    if isempty(vals1) || isempty(vals3)
        p_vals(b) = NaN;
    else
        p_vals(b) = ranksum(vals1, vals3);
    end

    med1(b) = median(vals1, 'omitnan');
    med3(b) = median(vals3, 'omitnan');
    waveBinCenters(b) = mean(wavenumbers_roi(idx));
end

p_vals(p_vals==0) = realmin;

% Multiple testing correction (FDR)
p_fdr = mafdr(p_vals, 'BHFDR', true);
significant = p_fdr < 0.05;
[~,sortedIdx] = sort(p_vals, 'ascend');

%% MWU Significance Plot
figure('Name','MWU Significance');
stem(waveBinCenters, -log10(p_vals), 'filled'); hold on;
yline(-log10(0.05), '--r', 'p=0.05');
xlabel('Wavenumber (cm^{-1})');
ylabel('-log_{10}(p-value)');
title('Binned MWU Test');
xlim([950 1800]); set(gca,'XDir','reverse');

for i = 1:min(topNToDisplay, numel(p_vals))
    idx = sortedIdx(i);
    if significant(idx)
        text(waveBinCenters(idx), -log10(p_vals(idx))+0.5, '*', 'FontSize',12,'HorizontalAlignment','center');
    end
    text(waveBinCenters(idx), -log10(p_vals(idx))+0.2, sprintf('%.1f', waveBinCenters(idx)), 'FontSize',8, 'HorizontalAlignment','center');
end

%% Mean Spectra Plot
figure('Name','Mean Spectra','Position',[100 100 800 500]);
tiledlayout(1,2);

mean1 = mean(X_flat_mean(idxWHO1,:),1); std1 = std(X_flat_mean(idxWHO1,:),0,1);
mean3 = mean(X_flat_mean(idxWHO3,:),1); std3 = std(X_flat_mean(idxWHO3,:),0,1);

nexttile;
fill([wavenumbers_roi fliplr(wavenumbers_roi)], [mean1-std1 fliplr(mean1+std1)], ...
     colors{1}, 'FaceAlpha', 0.2, 'EdgeColor','none'); hold on;
plot(wavenumbers_roi, mean1, 'Color', colors{1}, 'LineWidth',2);
title('WHO-1'); set(gca,'XDir','reverse'); xlim([950 1800]);
xlabel('Wavenumber (cm^{-1})'); ylabel('Absorbance (A.U.)');

nexttile;
fill([wavenumbers_roi fliplr(wavenumbers_roi)], [mean3-std3 fliplr(mean3+std3)], ...
     colors{2}, 'FaceAlpha', 0.2, 'EdgeColor','none'); hold on;
plot(wavenumbers_roi, mean3, 'Color', colors{2}, 'LineWidth',2);
title('WHO-3'); set(gca,'XDir','reverse'); xlim([950 1800]);
xlabel('Wavenumber (cm^{-1})'); ylabel('Absorbance (A.U.)');

%% Heatmap (individual spectra)
[X_flat, ~] = flatten_spectra_for_pca(analyzeTable, length(wavenumbers_roi));
X_flat = X_flat(:, roiMask);
X_flat = X_flat(validRows,:); % reorder to match filtered data
X_flat = [X_flat(idxWHO1,:); X_flat(idxWHO3,:)];

figure('Name','Spectral Heatmap');
imagesc(wavenumbers_roi, 1:size(X_flat,1), X_flat);
set(gca,'XDir','reverse'); colormap parula; colorbar;
xlabel('Wavenumber (cm^{-1})'); ylabel('Spectrum Index');
title('All Spectra Heatmap');
yline(sum(idxWHO1), 'k--', 'LineWidth',1.2);
text(1820, sum(idxWHO1)/2, 'WHO-1', 'FontSize',10,'Rotation',90);
text(1820, sum(idxWHO1)+(sum(idxWHO3)/2), 'WHO-3', 'FontSize',10,'Rotation',90);
yline(sum(idxWHO1), 'k--', 'LineWidth',1.2);
text(1820, sum(idxWHO1)/2, 'WHO-1', 'FontSize',10,'Rotation',90);
text(1820, sum(idxWHO1)+(sum(idxWHO3)/2), 'WHO-3', 'FontSize',10,'Rotation',90);
yline(sum(idxWHO1), 'k--', 'LineWidth',1.2);
text(1820, sum(idxWHO1)/2, 'WHO-1', 'FontSize',10,'Rotation',90);
text(1820, sum(idxWHO1)+(sum(idxWHO3)/2), 'WHO-3', 'FontSize',10,'Rotation',90);


%% Outlier Detection in PCA Space (PC1–PC3)
scoresUsed = score(:,1:3);  % Verwende PC1–3 für Distanzberechnung
mahalD = mahal(scoresUsed, scoresUsed);

% Top 3 Ausreißer nach Mahalanobis-Distanz
[sortedD, sortedIdx] = sort(mahalD, 'descend');
topOutliers = sortedIdx(1:3);

% Zeige deren Patient_ID oder Index
if ismember('Patient_ID', analyzeTable.Properties.VariableNames)
    disp('Top PCA Outliers (by Patient_ID):');
    disp(analyzeTable.Patient_ID(validRows(topOutliers)));
else
    disp('Top PCA Outliers (row indices):');
    disp(validRows(topOutliers));
end

% Plotte deren Spektren
figure('Name','Top PCA Outlier Spectra');
hold on;
for i = 1:length(topOutliers)
    plot(wavenumbers_roi, X_flat_mean(topOutliers(i),:), 'LineWidth', 2);
end
set(gca, 'XDir', 'reverse');
xlabel('Wavenumber (cm^{-1})'); ylabel('Absorbance');
title('Top Outlier Spectra in PCA Space');
legend(arrayfun(@(i) sprintf('Outlier %d', i), 1:length(topOutliers), 'UniformOutput', false));

% Optional: Farbiger PCA-Scatter nach Mahalanobis-Distanz
figure('Name','PCA Scatter colored by Mahalanobis Distance');
scatter(score(:,1), score(:,2), 40, mahalD, 'filled');
colorbar;
xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
title('Mahalanobis Distance in PCA Space');
grid on;

%% End
disp('Exploratory PCA & MWU analysis completed successfully.');