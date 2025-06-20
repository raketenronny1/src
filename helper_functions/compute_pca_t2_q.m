%%
% compute_pca_t2_q.m
%
% Perform PCA on a spectral matrix and calculate Hotelling T^2 and
% Q-statistic thresholds and flags.
%
% INPUTS:
%   X               - (N x D) matrix of spectra.
%   alpha           - Significance level for thresholding.
%   varianceToModel - Fraction of variance to explain when selecting the
%                     number of PCs for the T2/Q model.
%
% OUTPUT:
%   results - Struct with fields:
%       coeff, score, latent, explained, mu
%       k_model
%       T2_values, Q_values
%       T2_threshold, Q_threshold
%       flag_T2, flag_Q, is_T2_only, is_Q_only,
%       is_T2_and_Q, is_OR_outlier, is_normal
%
% Date: 2025-05-18

function results = compute_pca_t2_q(X, alpha, varianceToModel)

    if isempty(X) || size(X,1) < 2
        error('compute_pca_t2_q: Not enough data for PCA.');
    end

    if any(isnan(X(:))) || any(isinf(X(:)))
        X = fillmissing(X, 'mean');
        rows_with_inf = any(isinf(X),2);
        if any(rows_with_inf)
            X(rows_with_inf,:) = [];
            warning('compute_pca_t2_q: Removed %d rows containing Inf.', sum(rows_with_inf));
        end
    end

    [coeff, score, latent, ~, explained, mu] = pca(X, 'Algorithm','svd');
    if isempty(explained)
        error('compute_pca_t2_q: PCA returned empty results.');
    end

    cumulativeVariance = cumsum(explained);
    k_model = find(cumulativeVariance >= varianceToModel*100, 1, 'first');
    if isempty(k_model)
        k_model = length(explained);
    end
    if k_model == 0
        k_model = 1;
    end

    k_model = min(k_model, size(score,2));
    k_model = min(k_model, length(latent));

    tempScore = score(:,1:k_model);
    tempLambda = latent(1:k_model);
    tempLambda(tempLambda <= eps) = eps;
    T2_values = sum(bsxfun(@rdivide, tempScore.^2, tempLambda'),2);

    n_samples = size(X,1);
    if n_samples > k_model && k_model > 0
        T2_threshold = ((k_model*(n_samples-1))/(n_samples-k_model))*finv(1-alpha,k_model,n_samples-k_model);
    else
        T2_threshold = chi2inv(1-alpha,k_model);
    end

    k_for_recon = min(k_model, size(coeff,2));
    X_recon = score(:,1:k_for_recon) * coeff(:,1:k_for_recon)' + mu;
    Q_values = sum((X - X_recon).^2,2);

    num_total_actual_pcs = find(latent > eps,1,'last');
    if isempty(num_total_actual_pcs)
        num_total_actual_pcs = 0;
    end
    Q_threshold = NaN;
    if k_model < num_total_actual_pcs
        discarded = latent(k_model+1:num_total_actual_pcs);
        discarded(discarded <= eps) = eps;
        if ~isempty(discarded)
            th1 = sum(discarded); th2 = sum(discarded.^2); th3 = sum(discarded.^3);
            if th1 > eps && th2 > eps
                h0 = 1 - (2*th1*th3)/(3*th2^2);
                if h0 <= eps || isnan(h0) || isinf(h0)
                    h0 = 1;
                end
                ca = norminv(1-alpha);
                val = ca*sqrt(2*th2*h0^2)/th1 + 1 + th2*h0*(h0-1)/(th1^2);
                if val > 0 && h0 > 0
                    Q_threshold = th1 * (val)^(1/h0);
                end
            end
        end
        if isnan(Q_threshold) || isinf(Q_threshold)
            Q_threshold = prctile(Q_values,(1-alpha)*100);
        end
    else
        Q_threshold = 0;
    end

    flag_T2 = (T2_values > T2_threshold);
    flag_Q  = (Q_values > Q_threshold);
    is_T2_only  = flag_T2 & ~flag_Q;
    is_Q_only   = ~flag_T2 & flag_Q;
    is_T2_and_Q = flag_T2 & flag_Q;
    is_OR_outlier = flag_T2 | flag_Q;
    is_normal = ~is_OR_outlier;

    results = struct('coeff',coeff,'score',score,'latent',latent,'explained',explained,'mu',mu,
                     'k_model',k_model,'T2_values',T2_values,'Q_values',Q_values,
                     'T2_threshold',T2_threshold,'Q_threshold',Q_threshold,
                     'flag_T2',flag_T2,'flag_Q',flag_Q,'is_T2_only',is_T2_only,
                     'is_Q_only',is_Q_only,'is_T2_and_Q',is_T2_and_Q,
                     'is_OR_outlier',is_OR_outlier,'is_normal',is_normal);
end
