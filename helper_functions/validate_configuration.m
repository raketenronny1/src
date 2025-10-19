function cfg = validate_configuration(cfg)
%VALIDATE_CONFIGURATION Validate configuration struct for project scripts.
%
%   cfg = VALIDATE_CONFIGURATION(cfg) checks that required fields are present
%   and conform to expected types and value ranges. The helper normalises
%   string inputs, ensures referenced directories exist, and raises errors for
%   invalid settings. When deprecated configuration fields are encountered the
%   values are mapped onto their replacements and a warning is emitted via the
%   logging framework (if available).
%
%   A configuration struct can optionally contain a `logger` field referencing
%   an object or struct that exposes a `warn` or `warning` method/function. The
%   helper will delegate deprecated-field messages to this logger; otherwise it
%   falls back to MATLAB's `warning` function.
%
%   The following validations are applied:
%     * `projectRoot` must reference an existing folder.
%     * The `data` directory under the project root must exist.
%     * `useOutlierRemoval` and `parallelOutlierComparison` must be logical
%       scalars (numeric 0/1 are accepted and converted).
%     * `outlierAlpha` must be a numeric scalar in the open interval (0, 1).
%     * `outlierVarianceToModel` must be a numeric scalar in (0, 1].
%
%   Deprecated aliases handled by this helper:
%     * `useFilteredTrainingData` -> `useOutlierRemoval`
%     * `compareFilteredAndFullData` -> `parallelOutlierComparison`
%     * `outlierVarianceThreshold` -> `outlierVarianceToModel`
%
%   The helper returns the (potentially updated) configuration struct.
%
%   See also CONFIGURE_CFG, SETUP_PROJECT_PATHS.
%
% Copyright 2025.

    arguments
        cfg (1,1) struct
    end

    logger = [];
    if isfield(cfg, 'logger')
        logger = cfg.logger;
    end

    % ------------------------------------------------------------------
    % Handle deprecated fields
    % ------------------------------------------------------------------
    deprecatedMap = {
        'useFilteredTrainingData',    'useOutlierRemoval';
        'compareFilteredAndFullData', 'parallelOutlierComparison';
        'outlierVarianceThreshold',   'outlierVarianceToModel'
    };

    for i = 1:size(deprecatedMap, 1)
        oldField = deprecatedMap{i,1};
        newField = deprecatedMap{i,2};
        if isfield(cfg, oldField)
            if ~isfield(cfg, newField) || isempty(cfg.(newField))
                cfg.(newField) = cfg.(oldField);
            end
            emit_warning(logger, sprintf('Configuration field "%s" is deprecated. Please use "%s" instead.', oldField, newField), ...
                'validate_configuration:DeprecatedField');
            cfg = rmfield(cfg, oldField);
        end
    end

    % ------------------------------------------------------------------
    % Validate project root path
    % ------------------------------------------------------------------
    if ~isfield(cfg, 'projectRoot') || isempty(cfg.projectRoot)
        error('validate_configuration:MissingProjectRoot', ...
            'Configuration must define a projectRoot directory.');
    end

    projectRoot = normalise_text(cfg.projectRoot);
    if ~isfolder(projectRoot)
        error('validate_configuration:InvalidProjectRoot', ...
            'The configured projectRoot "%s" does not exist.', projectRoot);
    end
    cfg.projectRoot = projectRoot;

    dataDir = fullfile(projectRoot, 'data');
    if ~isfolder(dataDir)
        emit_warning(logger, sprintf('Expected data directory not found at %s. Creating it automatically.', dataDir), ...
            'validate_configuration:MissingDataDirectory');
        try
            mkdir(dataDir);
        catch ME
            warning('validate_configuration:DataDirectoryCreationFailed', ...
                'Unable to create data directory at %s: %s', dataDir, ME.message);
        end
    end

    % Warn if optional output folders are missing (they will be created later)
    optionalDirs = {'results','models','figures'};
    for i = 1:numel(optionalDirs)
        target = fullfile(projectRoot, optionalDirs{i});
        if ~isfolder(target)
            emit_warning(logger, sprintf('Optional project directory missing: %s (it will be created as needed).', target), ...
                'validate_configuration:MissingOptionalDirectory');
        end
    end

    % ------------------------------------------------------------------
    % Validate logical flags
    % ------------------------------------------------------------------
    logicalFields = {'useOutlierRemoval','parallelOutlierComparison'};
    for i = 1:numel(logicalFields)
        fieldName = logicalFields{i};
        if isfield(cfg, fieldName)
            value = cfg.(fieldName);
            if islogical(value) && isscalar(value)
                continue;
            elseif isnumeric(value) && isscalar(value)
                cfg.(fieldName) = logical(value);
            else
                error('validate_configuration:InvalidLogical', ...
                    'Configuration field "%s" must be a logical scalar.', fieldName);
            end
        end
    end

    % ------------------------------------------------------------------
    % Validate numeric parameters
    % ------------------------------------------------------------------
    if isfield(cfg, 'outlierAlpha')
        try
            validateattributes(cfg.outlierAlpha, {'numeric'}, {'scalar','real','>',0,'<',1}, mfilename, 'outlierAlpha');
        catch ME
            rethrow(maybe_wrap_identifier(ME, 'validate_configuration:InvalidOutlierAlpha'));
        end
    end

    if isfield(cfg, 'outlierVarianceToModel')
        try
            validateattributes(cfg.outlierVarianceToModel, {'numeric'}, {'scalar','real','>',0,'<=',1}, mfilename, 'outlierVarianceToModel');
        catch ME
            rethrow(maybe_wrap_identifier(ME, 'validate_configuration:InvalidOutlierVarianceToModel'));
        end
    end
end

function emit_warning(logger, message, identifier)
    if nargin < 3 || isempty(identifier)
        identifier = 'validate_configuration:Warning';
    end

    if isempty(logger)
        warning(identifier, '%s', message);
        return;
    end

    try
        if isa(logger, 'function_handle')
            logger('warning', message);
        elseif isobject(logger)
            if ismethod(logger, 'warn')
                logger.warn(message);
            elseif ismethod(logger, 'warning')
                logger.warning(message);
            elseif ismethod(logger, 'log')
                logger.log('warning', message);
            else
                warning(identifier, '%s', message);
            end
        elseif isstruct(logger)
            if isfield(logger, 'warn') && isa(logger.warn, 'function_handle')
                logger.warn(message);
            elseif isfield(logger, 'warning') && isa(logger.warning, 'function_handle')
                logger.warning(message);
            elseif isfield(logger, 'log') && isa(logger.log, 'function_handle')
                logger.log('warning', message);
            else
                warning(identifier, '%s', message);
            end
        else
            warning(identifier, '%s', message);
        end
    catch
        warning(identifier, '%s', message);
    end
end

function textOut = normalise_text(value)
    if isstring(value) && isscalar(value)
        textOut = char(value);
    elseif ischar(value)
        textOut = value;
    else
        error('validate_configuration:InvalidTextValue', ...
            'Configuration values must be character vectors or strings.');
    end
end

function MEout = maybe_wrap_identifier(ME, newIdentifier)
    if startsWith(ME.identifier, 'MATLAB:')
        MEout = MException(newIdentifier, '%s', ME.message);
        MEout = addCause(MEout, ME);
    else
        MEout = ME;
    end
end
