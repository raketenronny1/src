function entry = log_pipeline_message(level, context, message, varargin)
%LOG_PIPELINE_MESSAGE Unified logging utility for pipeline execution.
%   ENTRY = LOG_PIPELINE_MESSAGE(LEVEL, CONTEXT, MESSAGE, VARARGIN) formats
%   the MESSAGE with the supplied VARARGIN and writes it to the console with
%   a timestamp and severity LEVEL ("info", "warning" or "error"). CONTEXT
%   should describe the caller (e.g., "perform_inner_cv:fold1"). The
%   returned ENTRY struct captures the timestamp, level, context and
%   formatted message for optional downstream diagnostics.
%
%   The logger uses fprintf for informational output and fprintf(2, ...) for
%   warnings and errors so that the messages appear on the MATLAB command
%   window as they are emitted.
%
%   Date: 2025-07-06

    if nargin < 2 || isempty(context)
        context = 'pipeline';
    end
    if nargin < 1 || isempty(level)
        level = 'info';
    end
    if nargin < 3 || isempty(message)
        message = '';
    end

    levelLower = lower(string(level));
    if ~ismember(levelLower, ["info","warning","error"])
        levelLower = "info";
    end
    timestamp = datetime('now','Format','yyyy-MM-dd HH:mm:ss');
    formattedMessage = sprintf(message, varargin{:});
    prefix = sprintf('%s [%s] %s', char(timestamp), upper(levelLower), context);
    fullMessage = sprintf('%s %s\n', prefix, formattedMessage);

    switch levelLower
        case "info"
            fprintf('%s', fullMessage);
        case "warning"
            fprintf(2, '%s', fullMessage);
        case "error"
            fprintf(2, '%s', fullMessage);
    end

    entry = struct('timestamp', timestamp, ...
                   'level', upper(levelLower), ...
                   'context', context, ...
                   'message', formattedMessage);
end
