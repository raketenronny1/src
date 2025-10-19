%%
% flatten_spectra_for_pca.m
%
% Helper function to flatten spectra stored in a table for PCA and
% outlier analysis. Assumes the table has columns 'CombinedSpectra',
% 'WHO_Grade', and 'Diss_ID'.
%
% INPUTS:
%   dataTable           - Table with each row representing a probe. Each
%                          row must contain a numeric matrix in the
%                          'CombinedSpectra' column.
%   numWavenumberPoints - Expected number of features (wavenumbers) in
%                          each spectrum.
%
% OUTPUTS:
%   X_flat                           - Matrix of all spectra stacked
%                                      vertically.
%   y_cat_flat, y_numeric_flat       - Categorical and numeric labels for
%                                      each spectrum.
%   patientID_flat                   - Patient ID for each spectrum.
%   originalProbeRowIdx_flat         - Row index in dataTable for each
%                                      spectrum.
%   originalSpectrumIndexInProbe_flat- Index of the spectrum within its
%                                      original probe.
%
% Date: 2025-05-18

function [X_flat, y_numeric_flat, y_cat_flat, patientID_flat, ...
          originalProbeRowIdx_flat, originalSpectrumIndexInProbe_flat] = ...
          flatten_spectra_for_pca(dataTable, numWavenumberPoints)

    X_flat = [];
    y_cat_flat = categorical();
    patientID_flat = strings(0,1);
    originalProbeRowIdx_flat = [];
    originalSpectrumIndexInProbe_flat = [];

    if height(dataTable) == 0
        warning('flatten_spectra_for_pca: empty input table.');
        y_numeric_flat = [];
        return;
    end

    allSpectra_cell = {};
    allLabels_cell = {};
    patient_cell = {};
    probeRows = [];
    spectrumIdxInProbe = [];

    for i = 1:height(dataTable)
        spectraMatrix = dataTable.CombinedSpectra{i};
        if isempty(spectraMatrix) || ~isnumeric(spectraMatrix) || ndims(spectraMatrix) ~= 2
            continue;
        end
        if size(spectraMatrix,2) ~= numWavenumberPoints
            warning('flatten_spectra_for_pca: row %d wavenumber mismatch. Skipping.', i);
            continue;
        end
        numSpectra = size(spectraMatrix,1);
        allSpectra_cell{end+1,1} = spectraMatrix; %#ok<*AGROW>
        allLabels_cell{end+1,1} = repmat(dataTable.WHO_Grade(i), numSpectra,1);
        patient_cell{end+1,1} = repmat(dataTable.Diss_ID(i), numSpectra,1);
        probeRows = [probeRows; repmat(i, numSpectra,1)];
        spectrumIdxInProbe = [spectrumIdxInProbe; (1:numSpectra)'];
    end

    if isempty(allSpectra_cell)
        error('flatten_spectra_for_pca:NoValidSpectra', ...
              ['No valid spectra extracted. Troubleshooting tip: confirm the CombinedSpectra ', ...
               'column contains numeric matrices with %d columns and that preprocessing ', ...
               'scripts completed successfully.'], numWavenumberPoints);
    end

    X_flat = cell2mat(allSpectra_cell);
    y_cat_flat = cat(1, allLabels_cell{:});
    patientID_flat = vertcat(patient_cell{:});
    originalProbeRowIdx_flat = probeRows;
    originalSpectrumIndexInProbe_flat = spectrumIdxInProbe;

    % numeric labels
    y_numeric_flat = zeros(length(y_cat_flat),1);
    cats = categories(y_cat_flat);
    idx1 = find(strcmp(cats,'WHO-1'));
    idx3 = find(strcmp(cats,'WHO-3'));
    if ~isempty(idx1)
        y_numeric_flat(y_cat_flat == cats{idx1}) = 1;
    end
    if ~isempty(idx3)
        y_numeric_flat(y_cat_flat == cats{idx3}) = 3;
    end
end
