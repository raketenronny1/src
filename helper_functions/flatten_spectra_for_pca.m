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
% Optional name-value arguments:
%   'ChunkSize' - Number of table rows to process per batch. Smaller
%                 batches lower peak memory requirements at the cost of
%                 additional loop overhead (default [] -> all rows).
%
% Date: 2025-05-18

function [X_flat, y_numeric_flat, y_cat_flat, patientID_flat, ...
          originalProbeRowIdx_flat, originalSpectrumIndexInProbe_flat] = ...
          flatten_spectra_for_pca(dataTable, numWavenumberPoints, varargin)

    chunkSize = [];
    if ~isempty(varargin)
        if mod(numel(varargin),2) ~= 0
            error('flatten_spectra_for_pca:InvalidNameValue', 'Name-value arguments must occur in pairs.');
        end
        for nvIdx = 1:2:numel(varargin)
            name = lower(string(varargin{nvIdx}));
            value = varargin{nvIdx+1};
            switch name
                case "chunksize"
                    chunkSize = value;
                otherwise
                    error('flatten_spectra_for_pca:UnknownOption', 'Unknown option "%s".', name);
            end
        end
    end

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

    dataHeight = height(dataTable);
    if isempty(chunkSize) || ~isfinite(chunkSize) || chunkSize <= 0
        chunkSize = dataHeight;
    else
        chunkSize = min(dataHeight, max(1, floor(chunkSize)));
    end

    spectraBlocks = {};
    labelBlocks = {};
    patientBlocks = {};
    probeRowBlocks = {};
    spectrumIndexBlocks = {};

    for rowStart = 1:chunkSize:dataHeight
        rowEnd = min(rowStart + chunkSize - 1, dataHeight);
        chunkSpectraCell = {};
        chunkLabelCell = {};
        chunkPatientCell = {};
        chunkProbeRows = [];
        chunkSpectrumIdx = [];

        for i = rowStart:rowEnd
            spectraMatrix = dataTable.CombinedSpectra{i};
            if isempty(spectraMatrix) || ~isnumeric(spectraMatrix) || ndims(spectraMatrix) ~= 2
                continue;
            end
            if size(spectraMatrix,2) ~= numWavenumberPoints
                warning('flatten_spectra_for_pca: row %d wavenumber mismatch. Skipping.', i);
                continue;
            end
            numSpectra = size(spectraMatrix,1);
            chunkSpectraCell{end+1,1} = spectraMatrix; %#ok<AGROW>
            labelsStr = repmat(string(dataTable.WHO_Grade(i)), numSpectra,1);
            chunkLabelCell{end+1,1} = labelsStr;
            chunkPatientCell{end+1,1} = repmat(string(dataTable.Diss_ID(i)), numSpectra,1);
            chunkProbeRows = [chunkProbeRows; repmat(i, numSpectra,1)]; %#ok<AGROW>
            chunkSpectrumIdx = [chunkSpectrumIdx; (1:numSpectra)']; %#ok<AGROW>
        end

        if isempty(chunkSpectraCell)
            continue;
        end

        spectraBlocks{end+1,1} = vertcat(chunkSpectraCell{:}); %#ok<AGROW>
        labelBlocks{end+1,1} = vertcat(chunkLabelCell{:}); %#ok<AGROW>
        patientBlocks{end+1,1} = vertcat(chunkPatientCell{:}); %#ok<AGROW>
        probeRowBlocks{end+1,1} = chunkProbeRows; %#ok<AGROW>
        spectrumIndexBlocks{end+1,1} = chunkSpectrumIdx; %#ok<AGROW>
    end

    if isempty(spectraBlocks)
        error('flatten_spectra_for_pca:NoValidSpectra', ...
              ['No valid spectra extracted. Troubleshooting tip: confirm the CombinedSpectra ', ...
               'column contains numeric matrices with %d columns and that preprocessing ', ...
               'scripts completed successfully.'], numWavenumberPoints);
    end

    X_flat = vertcat(spectraBlocks{:});
    labelStrings = vertcat(labelBlocks{:});
    patientID_flat = vertcat(patientBlocks{:});
    originalProbeRowIdx_flat = vertcat(probeRowBlocks{:});
    originalSpectrumIndexInProbe_flat = vertcat(spectrumIndexBlocks{:});

    y_cat_flat = categorical(labelStrings);

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
