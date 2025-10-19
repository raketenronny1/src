function logger = create_parallel_logger(enableParallel)
%CREATE_PARALLEL_LOGGER Logging helper that works inside PARFOR loops.
%   LOGGER = CREATE_PARALLEL_LOGGER(ENABLEPARALLEL) returns a function handle
%   LOGGER(FMT, ...) that formats the message using SPRINTF and prints it with
%   a trailing newline. When ENABLEPARALLEL is true and parallel.pool.DataQueue
%   is available, messages are funnelled through a DataQueue to preserve order
%   across workers. Otherwise the logger falls back to fprintf.
%
%   Date: 2025-07-07

    if nargin < 1
        enableParallel = false;
    end

    useDataQueue = enableParallel && ...
        exist('parallel.pool.DataQueue','class') == 8 && ...
        exist('afterEach','file') == 2;

    if useDataQueue
        dq = parallel.pool.DataQueue; %#ok<TNMLP>
        afterEach(dq, @(msg) fprintf('%s\n', msg));
        logger = @(varargin) send(dq, sprintf(varargin{:}));
    else
        logger = @(varargin) fprintf('%s\n', sprintf(varargin{:}));
    end
end
