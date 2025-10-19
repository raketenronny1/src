function logger = setup_logging(cfg, sessionName)
%SETUP_LOGGING Initialize console and file logging for project scripts.
%   LOGGER = SETUP_LOGGING(CFG, SESSIONNAME) configures global logging
%   behaviour based on the provided configuration struct. The helper keeps a
%   global state that is consumed by LOG_MESSAGE. Call the returned
%   function handle LOGGER.closeFcn() when the script finishes to ensure the
%   log file is closed.
%
%   Supported configuration fields (either in CFG.logging or CFG itself):
%       projectRoot   - Base project directory. Defaults to pwd.
%       logDir        - Directory for log files. Defaults to
%                       fullfile(projectRoot,'results','Logs').
%       logFileName   - File name to use. Defaults to
%                       sprintf('%%s_%%s.log', datestr(now,'yyyymmdd_HHMMSS'),
%                       sessionName).
%       consoleLevel  - Minimum level to emit to the console (string or
%                       numeric). Defaults to 'info'.
%       fileLevel     - Minimum level to persist to the log file. Defaults to
%                       the console level.
%       autoFlush     - Logical flag to flush the log file after each write.
%
%   Levels (from most to least severe): 'error', 'warning', 'info', 'debug'.
%
%   The returned LOGGER struct mirrors the global state and exposes a
%   closeFcn that can be used in an onCleanup handler.
%
%   Example:
%       logger = setup_logging(cfg, 'Phase2');
%       cleanupObj = onCleanup(@()logger.closeFcn());
%       log_message('info','Hello world');

    arguments
        cfg (1,1) struct = struct()
        sessionName (1,1) string = "Session"
    end

    global LOGGER_STATE
    LOGGER_STATE = struct('isConfigured', false);

    % Allow nested logging configuration.
    if isfield(cfg, 'logging') && isstruct(cfg.logging)
        logCfg = cfg.logging;
        parentCfg = cfg;
    else
        logCfg = cfg;
        parentCfg = cfg;
    end

    if ~isfield(logCfg, 'projectRoot')
        if isfield(parentCfg, 'projectRoot') && ~isempty(parentCfg.projectRoot)
            logCfg.projectRoot = parentCfg.projectRoot;
        else
            logCfg.projectRoot = pwd;
        end
    end

    % Determine log directory and ensure it exists.
    if ~isfield(logCfg, 'logDir') || isempty(logCfg.logDir)
        logCfg.logDir = fullfile(logCfg.projectRoot, 'results', 'Logs');
    end
    if ~isfolder(logCfg.logDir)
        mkdir(logCfg.logDir);
    end

    if sessionName == ""
        sessionName = "Session";
    end

    % Determine log file name.
    if ~isfield(logCfg, 'logFileName') || isempty(logCfg.logFileName)
        timeStamp = datestr(now, 'yyyymmdd_HHMMSS');
        logCfg.logFileName = sprintf('%s_%s.log', timeStamp, char(sessionName));
    end
    logFilePath = fullfile(logCfg.logDir, logCfg.logFileName);

    % Resolve console and file verbosity thresholds.
    consoleLevel = resolve_level(logCfg, 'consoleLevel', 'info');
    fileLevel = resolve_level(logCfg, 'fileLevel', consoleLevel);

    % Auto flush flag (defaults to false).
    autoFlush = false;
    if isfield(logCfg, 'autoFlush') && ~isempty(logCfg.autoFlush)
        autoFlush = logical(logCfg.autoFlush);
    end

    % Attempt to open the log file. If it fails we still continue with
    % console logging only.
    fileID = -1;
    [fileID, logFilePath] = open_log_file(logFilePath);

    % Build global logger state.
    LOGGER_STATE = struct();
    LOGGER_STATE.isConfigured = true;
    LOGGER_STATE.sessionName = char(sessionName);
    LOGGER_STATE.consoleLevel = consoleLevel;
    LOGGER_STATE.fileLevel = fileLevel;
    LOGGER_STATE.fileID = fileID;
    LOGGER_STATE.logFilePath = logFilePath;
    LOGGER_STATE.autoFlush = autoFlush;
    LOGGER_STATE.startTime = datetime('now');

    % Provide a close function that can be re-used.
    LOGGER_STATE.closeFcn = @close_logger;
    LOGGER_STATE.closed = false;

    % Return a copy for callers (avoiding exposing file identifiers
    % directly). The closeFcn references the nested function below.
    logger = struct();
    logger.consoleLevel = consoleLevel;
    logger.fileLevel = fileLevel;
    logger.logFilePath = logFilePath;
    logger.sessionName = char(sessionName);
    logger.closeFcn = @close_logger;

    if fileID > 0
        log_message('debug', 'Logging initialised. Writing to %s', logFilePath);
    else
        log_message('debug', 'Logging initialised without file output. Directory: %s', logCfg.logDir);
    end

    function close_logger()
        global LOGGER_STATE
        if isempty(LOGGER_STATE) || ~isfield(LOGGER_STATE, 'fileID')
            return
        end
        if LOGGER_STATE.closed
            return
        end
        if LOGGER_STATE.fileID > 0
            try %#ok<TRYNC>
                fclose(LOGGER_STATE.fileID);
            end
        end
        LOGGER_STATE.closed = true;
        LOGGER_STATE.fileID = -1;
        LOGGER_STATE.isConfigured = false;
    end
end

function levelValue = resolve_level(cfg, fieldName, defaultValue)
    levelMap = struct('error', 0, 'warning', 1, 'info', 2, 'debug', 3);
    if isfield(cfg, fieldName) && ~isempty(cfg.(fieldName))
        raw = cfg.(fieldName);
        if isnumeric(raw)
            levelValue = max(0, min(3, floor(raw(1))));
            return
        end
        if isstring(raw) || ischar(raw)
            key = lower(string(raw));
            if isfield(levelMap, char(key))
                levelValue = levelMap.(char(key));
                return
            end
        end
    end
    if ischar(defaultValue) || isstring(defaultValue)
        levelValue = levelMap.(char(lower(string(defaultValue))));
    else
        levelValue = max(0, min(3, floor(defaultValue)));
    end
end

function [fileID, resolvedPath] = open_log_file(pathCandidate)
    [resolvedPathDir, ~, ~] = fileparts(pathCandidate);
    if ~isempty(resolvedPathDir) && ~isfolder(resolvedPathDir)
        mkdir(resolvedPathDir);
    end
    [fileID, resolvedPath] = deal(-1, pathCandidate);
    try
        [fileID, msg] = fopen(pathCandidate, 'a');
        if fileID < 0
            log_message('warning', 'Could not open log file %s: %s. Continuing without file logging.', ...
                pathCandidate, msg);
        end
    catch ME %#ok<CTCH>
        log_message('warning', 'Could not open log file %s: %s. Continuing without file logging.', ...
            pathCandidate, ME.message);
        fileID = -1;
    end
end
