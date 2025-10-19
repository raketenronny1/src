function rngInfo = set_random_seed(seed, varargin)
%SET_RANDOM_SEED Configure MATLAB's RNG and log the action.
%   rngInfo = SET_RANDOM_SEED(seed) sets the random number generator to the
%   specified SEED using the default 'twister' algorithm. When SEED is empty
%   or not provided the generator is initialised via rng('shuffle').
%
%   rngInfo = SET_RANDOM_SEED(seed, 'Logger', logger, 'Context', ctx)
%   allows specifying a logger (function handle or struct with an INFO
%   method) and a textual CONTEXT that is included in the log message.
%
%   The returned struct rngInfo contains the requested seed, the applied
%   seed reported by MATLAB, the RNG type and full state. This can be stored
%   alongside results for reproducibility tracking.
%
%   Example:
%       info = set_random_seed(42, 'Context', 'Phase 2');
%       disp(info.appliedSeed);
%
%   See also rng.
%
% Date: 2025-06-09

    parser = inputParser();
    parser.FunctionName = 'set_random_seed';
    addRequired(parser, 'seed', @(x) isempty(x) || (isscalar(x) && isnumeric(x) && isfinite(x)));
    addParameter(parser, 'Logger', [], @(x) isempty(x) || isa(x,'function_handle') || isstruct(x));
    addParameter(parser, 'Context', '', @(x) isstring(x) || ischar(x));
    parse(parser, seed, varargin{:});

    logger = parser.Results.Logger;
    context = strtrim(string(parser.Results.Context));

    requestedSeed = parser.Results.seed;
    if isempty(requestedSeed)
        rng('shuffle');
        seedMethod = "shuffle";
    else
        validateattributes(requestedSeed, {'numeric'}, {'scalar','integer','nonnegative'});
        rng(requestedSeed, 'twister');
        seedMethod = "fixed";
    end

    stateStruct = rng();
    rngInfo = struct();
    rngInfo.method = seedMethod;
    rngInfo.requestedSeed = requestedSeed;
    rngInfo.appliedSeed = stateStruct.Seed;
    rngInfo.rngType = stateStruct.Type;
    rngInfo.state = stateStruct.State;
    rngInfo.timestamp = datetime('now');

    logMessage = compose_log_message(context, rngInfo);
    dispatch_log(logger, logMessage);
end

function msg = compose_log_message(context, rngInfo)
    if strlength(context) > 0
        prefix = sprintf('%s random seed', context);
    else
        prefix = 'Random seed';
    end

    if isempty(rngInfo.requestedSeed)
        msg = sprintf('%s initialised via rng(''shuffle'') -> seed %d (%s).', ...
            prefix, rngInfo.appliedSeed, rngInfo.rngType);
    else
        msg = sprintf('%s set to %d (%s).', prefix, rngInfo.appliedSeed, rngInfo.rngType);
    end
end

function dispatch_log(logger, message)
    if isempty(logger)
        fprintf('%s\n', message);
        return;
    end

    if isa(logger, 'function_handle')
        logger(message);
        return;
    end

    if isstruct(logger)
        if isfield(logger, 'info') && isa(logger.info, 'function_handle')
            logger.info(message);
            return;
        elseif isfield(logger, 'log') && isa(logger.log, 'function_handle')
            logger.log(message);
            return;
        end
    end

    fprintf('%s\n', message);
end
