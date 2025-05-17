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
% Date: 2025-05-15


function [spectra_binned, wavenumbers_binned] = bin_spectra(spectra, wavenumbers, binningFactor)

    if binningFactor <= 1 || isempty(spectra) || isempty(wavenumbers)
        spectra_binned = spectra;
        wavenumbers_binned = wavenumbers;
        return;
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

    for i = 1:numBinnedFeatures
        startIdx = (i-1) * binningFactor + 1;
        endIdx = i * binningFactor;
        
        % This loop structure ensures endIdx will not exceed numOriginalFeatures
        % because numBinnedFeatures is calculated using floor().
        % The last few features (less than binningFactor) will be ignored.
        
        spectra_binned(:, i) = mean(spectra(:, startIdx:endIdx), 2);
        wavenumbers_binned(i) = mean(wavenumbers(startIdx:endIdx));
    end
end