function [seed, source] = resolve_random_seed(cfg, preferredField, fallbackField)
%RESOLVE_RANDOM_SEED Determine which seed value to use for reproducibility.
%   [SEED, SOURCE] = RESOLVE_RANDOM_SEED(CFG, PREFERRED, FALLBACK) inspects
%   the configuration struct CFG and returns the first non-empty numeric
%   seed found in the specified fields. SOURCE is the name of the field that
%   provided the seed, or "shuffle" when no explicit seed is configured.
%
%   The helper supports lightweight configuration files that only specify a
%   single global seed as well as phase-specific overrides. When neither the
%   preferred nor fallback field contains a value, the returned seed is
%   empty so callers can fall back to rng('shuffle').
%
%   Example:
%       [seed, source] = resolve_random_seed(cfg, 'randomSeedPhase2');
%
%   See also set_random_seed.

    if nargin < 2 || isempty(preferredField)
        preferredField = 'randomSeed';
    end
    if nargin < 3 || isempty(fallbackField)
        fallbackField = 'randomSeed';
    end

    seed = [];
    source = "shuffle";

    if ~isstruct(cfg)
        return;
    end

    if isfield(cfg, preferredField) && ~isempty(cfg.(preferredField))
        seedCandidate = cfg.(preferredField);
        if validate_seed_candidate(seedCandidate)
            seed = double(seedCandidate);
            source = preferredField;
            return;
        end
    end

    if isfield(cfg, fallbackField) && ~isempty(cfg.(fallbackField))
        seedCandidate = cfg.(fallbackField);
        if validate_seed_candidate(seedCandidate)
            seed = double(seedCandidate);
            source = fallbackField;
            return;
        end
    end
end

function tf = validate_seed_candidate(value)
    tf = isnumeric(value) && isscalar(value) && isfinite(value);
end
