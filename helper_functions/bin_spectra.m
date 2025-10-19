%% 
% 

% bin_spectra.m
%
% Helper function to bin spectra by averaging adjacent features.
%
% INPUTS:
%   spectra         - (N_spectra x N_features_original) matrix of spectral data.
%   wavenumbers     - (1 x N_features_original) vector of original wavenumbers.
%   binningFactor   - (scalar integer) Number of adjacent features to average.
%                     If 1 or less, no binning is performed.
%
% OUTPUTS:
%   spectra_binned    - (N_spectra x N_features_binned) matrix of binned spectra.
%   wavenumbers_binned- (1 x N_features_binned) vector of new mean wavenumbers for bins.
%
% Optional name-value arguments:
%   'ChunkSize'     - Maximum number of spectra rows processed per loop
%                     iteration. Use to limit memory usage when spectra is
%                     very large (default [] -> process all rows at once).
%
% Date: 2025-05-15


function [spectra_binned, wavenumbers_binned] = bin_spectra(spectra, wavenumbers, binningFactor, varargin)

    if binningFactor <= 1 || isempty(spectra) || isempty(wavenumbers)
        spectra_binned = spectra;
        wavenumbers_binned = wavenumbers;
        return;
    end

    chunkSize = [];
    if ~isempty(varargin)
        if mod(numel(varargin),2) ~= 0
            error('bin_spectra:InvalidNameValue', 'Name-value arguments must occur in pairs.');
        end
        for nvIdx = 1:2:numel(varargin)
            name = lower(string(varargin{nvIdx}));
            value = varargin{nvIdx+1};
            switch name
                case "chunksize"
                    chunkSize = value;
                otherwise
                    error('bin_spectra:UnknownOption', 'Unknown option "%s".', name);
            end
        end
    end

    [numSpectra, numOriginalFeatures] = size(spectra);
    
    % Ensure wavenumbers is a row vector for consistency
    if iscolumn(wavenumbers)
        wavenumbers = wavenumbers';
    end

    if numOriginalFeatures ~= length(wavenumbers)
        error('Number of spectral features must match the length of the wavenumbers vector.');
    end

    numBinnedFeatures = floor(numOriginalFeatures / binningFactor);

    if numBinnedFeatures == 0
        warning('Binning factor is larger than the number of features. Returning original spectra or empty if not possible.');
        % If binning factor > num features, effectively no bins can be formed
        % based on strict averaging of 'binningFactor' features.
        % Depending on desired behavior, could return original or empty.
        % Returning empty binned spectra/wavenumbers is safer to indicate issue.
        spectra_binned = spectra(:,1:0); % Empty matrix with correct number of rows
        wavenumbers_binned = wavenumbers(1:0); % Empty row vector
        return;
    end

    spectra_binned = zeros(numSpectra, numBinnedFeatures);
    wavenumbers_binned = zeros(1, numBinnedFeatures);

    binStartIdx = (0:numBinnedFeatures-1) * binningFactor + 1;
    binEndIdx = binStartIdx + binningFactor - 1;
    for i = 1:numBinnedFeatures
        wavenumbers_binned(i) = mean(wavenumbers(binStartIdx(i):binEndIdx(i)));
    end

    if isempty(chunkSize) || ~isfinite(chunkSize) || chunkSize <= 0
        chunkSize = numSpectra;
    else
        chunkSize = min(numSpectra, max(1, floor(chunkSize)));
    end

    if numSpectra == 0 || chunkSize == 0
        return;
    end

    for rowStart = 1:chunkSize:numSpectra
        rowEnd = min(rowStart + chunkSize - 1, numSpectra);
        chunk = spectra(rowStart:rowEnd, :);
        chunkBinned = zeros(size(chunk,1), numBinnedFeatures);
        for i = 1:numBinnedFeatures
            chunkBinned(:, i) = mean(chunk(:, binStartIdx(i):binEndIdx(i)), 2);
        end
        spectra_binned(rowStart:rowEnd, :) = chunkBinned;
    end
end
