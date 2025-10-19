function [X_processed, processedWavenumbers, preprocessInfo] = apply_pipeline_preprocessing(X, wavenumbers, hyperparams, preprocessInfo)
%APPLY_PIPELINE_PREPROCESSING Apply shared preprocessing steps for spectra.
%   [X_PROCESSED, WN_PROCESSED, INFO] = APPLY_PIPELINE_PREPROCESSING(X, WN,
%   HYPERPARAMS) bins spectra according to the provided hyperparameters and
%   returns the transformed spectra together with the updated wavenumbers
%   and a struct describing the preprocessing steps.
%
%   [..., INFO] = APPLY_PIPELINE_PREPROCESSING(..., INFO) reuses an existing
%   INFO struct (e.g., computed on a training fold) so that the exact same
%   preprocessing can be applied to validation or test data.
%
%   The returned INFO struct contains at least:
%       .binningFactor         - binning factor that was applied
%       .processedWavenumbers  - wavenumbers after binning
%
%   Date: 2025-07-07
%
%   This helper centralises the binning logic used by both the inner
%   cross-validation routine and the final pipeline training helper.

    if nargin < 3 || isempty(hyperparams)
        hyperparams = struct();
    end
    if nargin < 4
        preprocessInfo = struct();
    end

    if iscolumn(wavenumbers)
        wavenumbers = wavenumbers';
    end

    reuseInfo = ~isempty(fieldnames(preprocessInfo));
    if reuseInfo && (~isfield(preprocessInfo,'binningFactor') || ~isfield(preprocessInfo,'processedWavenumbers'))
        reuseInfo = false;
    end

    if reuseInfo
        binningFactor = preprocessInfo.binningFactor;
        processedWavenumbers = preprocessInfo.processedWavenumbers;
    else
        if isfield(hyperparams, 'binningFactor') && hyperparams.binningFactor > 1
            binningFactor = hyperparams.binningFactor;
        else
            binningFactor = 1;
        end
    end

    if binningFactor > 1
        [X_processed, processedWavenumbers_local] = bin_spectra(X, wavenumbers, binningFactor);
        if reuseInfo
            % Validate that the reused info matches what binning produced.
            tolDenom = max(1, max(abs(processedWavenumbers(:))));
            if ~isequal(size(processedWavenumbers_local), size(processedWavenumbers)) || ...
               any(abs(processedWavenumbers_local - processedWavenumbers) > 10*eps(tolDenom))
                warning('apply_pipeline_preprocessing:WavenumberMismatch', ...
                    'Recomputed binned wavenumbers differ from stored values. Using recomputed version.');
                processedWavenumbers = processedWavenumbers_local;
            end
        else
            processedWavenumbers = processedWavenumbers_local;
        end
        X_processed = X_processed;
    else
        X_processed = X;
        processedWavenumbers = wavenumbers;
    end

    if ~reuseInfo
        preprocessInfo = struct();
        preprocessInfo.binningFactor = binningFactor;
        preprocessInfo.processedWavenumbers = processedWavenumbers;
    end
end
