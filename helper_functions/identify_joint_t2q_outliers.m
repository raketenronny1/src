function outlierStruct = identify_joint_t2q_outliers(X, alpha, varianceToModel)
%IDENTIFY_JOINT_T2Q_OUTLIERS Flag spectra that are outliers for both
%Hotelling's T2 and the Q-statistic.
%
%   outlierStruct = IDENTIFY_JOINT_T2Q_OUTLIERS(X, alpha, varianceToModel)
%   performs PCA on the spectra matrix X and returns a struct containing the
%   PCA model together with logical masks for spectra that are flagged as
%   outliers simultaneously by Hotelling's T2 statistic and the Q-statistic.
%
%   INPUTS:
%       X               - (N x D) matrix of spectra where rows correspond to
%                         individual measurements.
%       alpha           - Significance level (default 0.01).
%       varianceToModel - Fraction of variance to explain when selecting the
%                         number of principal components (default 0.95).
%
%   OUTPUT:
%       outlierStruct - struct with fields:
%           .pcaModel         - output of compute_pca_t2_q.
%           .isJointOutlier   - logical vector where true indicates spectra
%                                flagged as outliers by both T2 and Q.
%           .isJointInlier    - logical vector indicating spectra retained.
%           .numJointOutliers - count of joint outliers.
%           .alpha            - alpha used.
%           .varianceToModel  - variance fraction used.
%
%   The helper wraps compute_pca_t2_q so that downstream scripts can reuse a
%   consistent joint-outlier definition without duplicating the logic.
%
% Date: 2025-06-28
%
% See also COMPUTE_PCA_T2_Q.

    arguments
        X double {mustBeNonempty}
        alpha (1,1) double {mustBeGreaterThan(alpha,0), mustBeLessThan(alpha,1)} = 0.01
        varianceToModel (1,1) double {mustBeGreaterThan(varianceToModel,0), mustBeLessThanOrEqual(varianceToModel,1)} = 0.95
    end

    pcaModel = compute_pca_t2_q(X, alpha, varianceToModel);
    isJointOutlier = pcaModel.flag_T2 & pcaModel.flag_Q;
    isJointInlier = ~isJointOutlier;

    outlierStruct = struct( ...
        'pcaModel', pcaModel, ...
        'isJointOutlier', isJointOutlier, ...
        'isJointInlier', isJointInlier, ...
        'numJointOutliers', sum(isJointOutlier), ...
        'alpha', alpha, ...
        'varianceToModel', varianceToModel);
end
