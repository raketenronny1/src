function chunkSize = get_cfg_chunk_size(cfg, fieldName, defaultValue)
%GET_CFG_CHUNK_SIZE Return configured chunk size for a processing stage.
%   chunkSize = GET_CFG_CHUNK_SIZE(cfg, fieldName) looks for cfg.chunkSizes.
%   chunkSize = GET_CFG_CHUNK_SIZE(cfg, fieldName, defaultValue) returns
%   defaultValue when the configuration or field is absent.
%
%   The helper normalises invalid or non-positive values to [].
%
%   Inputs:
%       cfg          - configuration struct that may contain chunkSizes.
%       fieldName    - name of the chunk size field inside cfg.chunkSizes.
%       defaultValue - optional fallback (defaults to []).
%
%   Output:
%       chunkSize    - numeric chunk size or [].
%
%   This helper centralises the logic used by the Phase scripts so that
%   chunk-aware helpers can share consistent defaults.
%
%   Date: 2025-07-06

    if nargin < 3
        defaultValue = [];
    end

    chunkSize = defaultValue;
    if ~isstruct(cfg) || ~isfield(cfg, 'chunkSizes') || ~isstruct(cfg.chunkSizes)
        return;
    end

    if isfield(cfg.chunkSizes, fieldName)
        candidate = cfg.chunkSizes.(fieldName);
        if isempty(candidate) || ~isnumeric(candidate)
            chunkSize = [];
        elseif ~isfinite(candidate) || candidate <= 0
            chunkSize = [];
        else
            chunkSize = candidate;
        end
    end
end
