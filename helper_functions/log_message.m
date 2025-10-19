function log_message(level, message, varargin)
%LOG_MESSAGE Emit a timestamped log line respecting configured verbosity.
%   LOG_MESSAGE(LEVEL, MESSAGE, VARARGIN) routes formatted output to the
%   console and optional log file. LEVEL accepts 'error', 'warning', 'info',
%   or 'debug'. MESSAGE is a sprintf-style template.
%
%   Example:
%       log_message('info', 'Processing fold %d of %d', k, nFolds);

    if nargin < 1 || isempty(level)
        level = 'info';
    end
    if nargin < 2
        message = '';
    end

    levelKey = normalise_level(level);
    if isempty(levelKey)
        levelKey = 'info';
    end
    severity = map_level_to_severity(levelKey);

    if ~isempty(varargin)
        try
            message = sprintf(message, varargin{:});
        catch
            message = sprintf('[log_message] Formatting failed for message "%s".', message);
            severity = map_level_to_severity('warning');
            levelKey = 'warning';
        end
    end

    message = strip(message, 'right');
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    logLine = sprintf('%s [%s] %s', timestamp, upper(levelKey), message);

    global LOGGER_STATE
    if isempty(LOGGER_STATE) || ~isfield(LOGGER_STATE, 'isConfigured') || ~LOGGER_STATE.isConfigured
        emit_console(severity, logLine);
        return
    end

    if severity <= LOGGER_STATE.consoleLevel
        emit_console(severity, logLine);
    end

    if LOGGER_STATE.fileID > 0 && severity <= LOGGER_STATE.fileLevel
        fprintf(LOGGER_STATE.fileID, '%s\n', logLine);
        if isfield(LOGGER_STATE, 'autoFlush') && LOGGER_STATE.autoFlush
            fflush(LOGGER_STATE.fileID);
        end
    end
end

function emit_console(severity, logLine)
    switch severity
        case 0 % error
            fprintf(2, '%s\n', logLine);
        case 1 % warning
            fprintf(2, '%s\n', logLine);
        otherwise % info/debug
            fprintf('%s\n', logLine);
    end
end

function severity = map_level_to_severity(levelKey)
    switch lower(levelKey)
        case 'error'
            severity = 0;
        case {'warn', 'warning'}
            severity = 1;
        case 'debug'
            severity = 3;
        otherwise
            severity = 2;
    end
end

function key = normalise_level(level)
    if isnumeric(level)
        switch max(0, min(3, floor(level(1))))
            case 0
                key = 'error';
            case 1
                key = 'warning';
            case 3
                key = 'debug';
            otherwise
                key = 'info';
        end
        return
    end
    levelStr = lower(string(level));
    valid = {'error','warning','warn','info','debug'};
    if any(strcmp(levelStr, valid))
        key = char(levelStr);
    else
        key = '';
    end
end
