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
% Date: 2025-05-15

function fisherRatios = calculate_fisher_ratio(X, y)

    classes = unique(y);
    if length(classes) ~= 2
        error('Fisher Ratio calculation requires exactly two classes. Found %d.', length(classes));
    end

    class1_indices = (y == classes(1));
    class2_indices = (y == classes(2));

    if sum(class1_indices) < 2 || sum(class2_indices) < 2
        warning('calculate_fisher_ratio: One or both classes have fewer than 2 samples. Variances might be NaN or 0. Fisher ratios may be Inf or NaN.');
        % Proceeding, but user should be aware if sample size per class is tiny.
    end
    
    X1 = X(class1_indices, :);
    X2 = X(class2_indices, :);

    if isempty(X1) || isempty(X2) % Should be caught by <2 check generally
        warning('calculate_fisher_ratio: One or both classes have no samples after indexing. Fisher ratios will be NaN.');
        fisherRatios = NaN(1, size(X,2));
        return;
    end

    mean1 = mean(X1, 1);
    mean2 = mean(X2, 1);
    
    % var(X,0,1) calculates variance using (N-1) denominator.
    % If a class has only 1 sample, var will be NaN.
    var1  = var(X1, 0, 1); 
    var2  = var(X2, 0, 1);

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