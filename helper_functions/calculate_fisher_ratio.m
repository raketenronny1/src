%% 
% 

% calculate_fisher_ratio.m
%
% Helper function to calculate Fisher Ratio for each feature.
% Fisher Ratio = (mean_class1 - mean_class2)^2 / (variance_class1 + variance_class2)
% Assumes y contains two unique class labels.
%
% INPUTS:
%   X - (N_spectra x N_features) matrix of spectral data.
%   y - (N_spectra x 1) vector of class labels.
%
% OUTPUTS:
%   fisherRatios - (1 x N_features) vector of Fisher ratios.
%
% Optional name-value arguments:
%   'ChunkSize' - Number of spectra rows per class to accumulate per
%                 iteration. Use smaller values to reduce peak memory
%                 usage when X is very large (default [] -> all rows).
%
% Date: 2025-05-15

function fisherRatios = calculate_fisher_ratio(X, y, varargin)

    chunkSize = [];
    if ~isempty(varargin)
        if mod(numel(varargin),2) ~= 0
            error('calculate_fisher_ratio:InvalidNameValue', 'Name-value arguments must occur in pairs.');
        end
        for nvIdx = 1:2:numel(varargin)
            name = lower(string(varargin{nvIdx}));
            value = varargin{nvIdx+1};
            switch name
                case "chunksize"
                    chunkSize = value;
                otherwise
                    error('calculate_fisher_ratio:UnknownOption', 'Unknown option "%s".', name);
            end
        end
    end

    classes = unique(y);
    if length(classes) ~= 2
        error('calculate_fisher_ratio:InvalidClassCount', ...
              ['Fisher Ratio calculation requires exactly two classes but found %d. ', ...
               'Troubleshooting tip: verify that label preprocessing produces a binary ', ...
               'outcome or adjust the feature_selection_method configuration.'], ...
              length(classes));
    end

    class1_indices = (y == classes(1));
    class2_indices = (y == classes(2));

    if sum(class1_indices) < 2 || sum(class2_indices) < 2
        warning('calculate_fisher_ratio: One or both classes have fewer than 2 samples. Variances might be NaN or 0. Fisher ratios may be Inf or NaN.');
        % Proceeding, but user should be aware if sample size per class is tiny.
    end
    
    [numSamples, numFeatures] = size(X);
    if isempty(chunkSize) || ~isfinite(chunkSize) || chunkSize <= 0
        chunkSize = numSamples;
    else
        chunkSize = min(numSamples, max(1, floor(chunkSize)));
    end

    if numSamples == 0 || chunkSize == 0
        warning('calculate_fisher_ratio: Empty data matrix provided. Returning NaNs.');
        fisherRatios = NaN(1, numFeatures);
        return;
    end

    sum1 = zeros(1, numFeatures);
    sum2 = zeros(1, numFeatures);
    sumsq1 = zeros(1, numFeatures);
    sumsq2 = zeros(1, numFeatures);
    count1 = 0;
    count2 = 0;

    for rowStart = 1:chunkSize:numSamples
        rowEnd = min(rowStart + chunkSize - 1, numSamples);
        chunkX = X(rowStart:rowEnd, :);
        chunkMask1 = class1_indices(rowStart:rowEnd);
        chunkMask2 = class2_indices(rowStart:rowEnd);

        if any(chunkMask1)
            X1_chunk = chunkX(chunkMask1, :);
            sum1 = sum1 + sum(X1_chunk, 1);
            sumsq1 = sumsq1 + sum(X1_chunk.^2, 1);
            count1 = count1 + size(X1_chunk,1);
        end
        if any(chunkMask2)
            X2_chunk = chunkX(chunkMask2, :);
            sum2 = sum2 + sum(X2_chunk, 1);
            sumsq2 = sumsq2 + sum(X2_chunk.^2, 1);
            count2 = count2 + size(X2_chunk,1);
        end
    end

    if count1 == 0 || count2 == 0
        warning('calculate_fisher_ratio: One or both classes have no samples after indexing. Fisher ratios will be NaN.');
        fisherRatios = NaN(1, numFeatures);
        return;
    end

    mean1 = sum1 / max(count1, 1);
    mean2 = sum2 / max(count2, 1);

    var1 = NaN(1, numFeatures);
    var2 = NaN(1, numFeatures);
    if count1 > 1
        var1 = (sumsq1 - (sum1.^2) / count1) / (count1 - 1);
    end
    if count2 > 1
        var2 = (sumsq2 - (sum2.^2) / count2) / (count2 - 1);
    end

    var1(var1 < 0) = 0; % numerical guard
    var2(var2 < 0) = 0;

    % If variance is NaN (e.g., only 1 sample in class), or 0 (all samples in class are identical for that feature)
    % replace NaN with 0 for denominator stability, assuming if var is NaN due to 1 sample, it's like 0 spread for that point.
    % This is a choice; alternative is to let ratio be NaN.
    var1(isnan(var1)) = 0; 
    var2(isnan(var2)) = 0;

    numerator = (mean1 - mean2).^2;
    denominator = var1 + var2;

    fisherRatios = numerator ./ denominator;
    
    % Handle specific cases for division by zero:
    % Case 1: Denominator is 0 AND Numerator is 0 (means are same, variances are zero) -> Ratio is 0 (no separation)
    fisherRatios(denominator == 0 & numerator == 0) = 0;
    % Case 2: Denominator is 0 AND Numerator is non-zero (means different, but variances zero) -> Ratio is Inf (perfect separation, potentially problematic)
    fisherRatios(denominator == 0 & numerator ~= 0) = Inf;
    % Case 3: If any variance was NaN and became 0, and other variance is also 0, leading to 0/0 if means were same.
    % If a var was NaN (single sample) and now 0, if mean diff exists, this could also lead to Inf.
end
