function cfg = configure_cfg(varargin)
%CONFIGURE_CFG Create or update configuration struct for project scripts.
%
%   cfg = CONFIGURE_CFG() loads config/default.yaml and returns the merged
%   configuration. Additional overrides are applied in the following order:
%     1. config/<ENV>.yaml if MENINGIOMA_ENV is set
%     2. File referenced by MENINGIOMA_CONFIG (if set)
%     3. Explicit 'configFile' name/value pair
%     4. Existing struct passed as the first argument
%     5. Remaining name/value arguments
%
%   Name/value pairs other than 'configFile' are treated as field overrides.
%
%   Example:
%       cfg = configure_cfg('configFile','config/custom.yaml', ...
%                           'useOutlierRemoval',false);
%
%   The returned struct exposes both the nested sections (paths, analysis,
%   exports) and the flattened legacy fields (e.g. cfg.projectRoot) to remain
%   compatible with existing scripts.

    if nargin > 0 && isstruct(varargin{1})
        initialCfg = varargin{1};
        args = varargin(2:end);
    else
        initialCfg = struct();
        args = varargin;
    end

    if mod(numel(args),2) ~= 0
        error('configure_cfg:NameValuePairs', ...
              ['Name-value arguments must occur in pairs. Troubleshooting tip: check configure_cfg ', ...
               'calls for a missing value or trailing parameter.']);
    end

    configFile = '';
    overrides = struct();
    for i = 1:2:numel(args)
        name = args{i};
        value = args{i+1};
        if strcmpi(name,'configFile')
            configFile = value;
        else
            overrides.(name) = value;
        end
    end

    repoRoot = resolve_project_root(initialCfg);
    cfg = struct();

    % Load default config
    defaultPath = fullfile(repoRoot,'config','default.yaml');
    if isfile(defaultPath)
        cfg = merge_structs(cfg, read_yaml_config(defaultPath));
    end

    % Optional MENINGIOMA_ENV override (config/<env>.yaml)
    envName = strtrim(getenv('MENINGIOMA_ENV'));
    if ~isempty(envName)
        envPath = fullfile(repoRoot,'config',[envName '.yaml']);
        if ~isfile(envPath)
            envPath = fullfile(repoRoot,'config',[lower(envName) '.yaml']);
        end
        if isfile(envPath)
            cfg = merge_structs(cfg, read_yaml_config(envPath));
        else
            warning('configure_cfg:MissingEnvOverride', ...
                'MENINGIOMA_ENV set to "%s" but no config file found.', envName);
        end
    end

    % Optional MENINGIOMA_CONFIG override (explicit path)
    envConfigPath = strtrim(getenv('MENINGIOMA_CONFIG'));
    if ~isempty(envConfigPath)
        resolved = resolve_config_path(envConfigPath, repoRoot);
        if isfile(resolved)
            cfg = merge_structs(cfg, read_yaml_config(resolved));
        else
            warning('configure_cfg:MissingEnvConfig', ...
                'MENINGIOMA_CONFIG points to "%s" but the file is missing.', envConfigPath);
        end
    end

    % Explicit configFile override
    if ~isempty(configFile)
        resolved = resolve_config_path(configFile, repoRoot);
        if ~isfile(resolved)
            error('configure_cfg:ConfigNotFound', ...
                'Configuration file not found: %s', resolved);
        end
        cfg = merge_structs(cfg, read_yaml_config(resolved));
    end

    % Merge struct argument then name/value overrides
    cfg = merge_structs(cfg, initialCfg);
    cfg = merge_structs(cfg, overrides);

    % Flatten nested sections for backwards compatibility
    cfg = flatten_sections(cfg, {'paths','analysis','exports'});

    % Resolve project root relative to repository
    if ~isfield(cfg,'projectRoot') || isempty(cfg.projectRoot)
        cfg.projectRoot = repoRoot;
    else
        cfg.projectRoot = resolve_config_path(cfg.projectRoot, repoRoot);
    end

    % Derive legacy defaults if still missing
    defaults = struct( ...
        'useOutlierRemoval', true, ...
        'parallelOutlierComparison', true, ...
        'outlierAlpha', 0.01, ...
        'outlierVarianceToModel', 0.95 ...
    );
    cfg = merge_missing(cfg, defaults);

    % Final validation pass before returning to callers
    cfg = validate_configuration(cfg);
end

function root = resolve_project_root(existingCfg)
    if isstruct(existingCfg) && isfield(existingCfg,'projectRoot') && ~isempty(existingCfg.projectRoot)
        root = existingCfg.projectRoot;
        return;
    end
    if exist('get_project_root','file')
        root = get_project_root();
    else
        root = pwd;
    end
end

function resolved = resolve_config_path(pathStr, basePath)
    if isstring(pathStr)
        pathStr = char(pathStr);
    end
    if startsWith(pathStr,'~')
        userDir = char(java.lang.System.getProperty('user.home'));
        resolved = fullfile(userDir, pathStr(2:end));
    elseif is_absolute_path(pathStr)
        resolved = pathStr;
    else
        resolved = fullfile(basePath, pathStr);
    end
    resolved = char(java.io.File(resolved).getPath());
end

function tf = is_absolute_path(pathStr)
    if isempty(pathStr)
        tf = false;
        return;
    end
    if startsWith(pathStr, filesep) || ...
            (~isempty(regexp(pathStr, '^[A-Za-z]:', 'once')))
        tf = true;
    else
        tf = false;
    end
end

function out = merge_structs(base, override)
    out = base;
    if ~isstruct(override)
        return;
    end
    fields = fieldnames(override);
    for i = 1:numel(fields)
        name = fields{i};
        value = override.(name);
        if isstruct(value) && isfield(out,name) && isstruct(out.(name))
            out.(name) = merge_structs(out.(name), value);
        else
            out.(name) = value;
        end
    end
end

function cfg = flatten_sections(cfg, sectionNames)
    for i = 1:numel(sectionNames)
        section = sectionNames{i};
        if isfield(cfg, section) && isstruct(cfg.(section))
            sectionStruct = cfg.(section);
            innerFields = fieldnames(sectionStruct);
            for j = 1:numel(innerFields)
                cfg.(innerFields{j}) = sectionStruct.(innerFields{j});
            end
        end
    end
end

function cfg = merge_missing(cfg, defaults)
    defFields = fieldnames(defaults);
    for i = 1:numel(defFields)
        name = defFields{i};
        if ~isfield(cfg,name) || isempty(cfg.(name))
            cfg.(name) = defaults.(name);
        end
    end
end

function data = read_yaml_config(filePath)
    text = fileread(filePath);
    lines = regexp(text, '\r?\n', 'split');
    data = struct();
    indentStack = -1;
    pathStack = { { } };

    for i = 1:numel(lines)
        line = lines{i};
        commentIdx = find(line == '#', 1);
        if ~isempty(commentIdx)
            line = line(1:commentIdx-1);
        end
        if all(isspace(line))
            continue;
        end

        indent = find(~isspace(line), 1) - 1;
        if isempty(indent)
            continue;
        end
        line = strtrim(line);

        while indent <= indentStack(end)
            indentStack(end) = [];
            pathStack(end) = [];
        end

        parts = strsplit(line, ':');
        key = strtrim(parts{1});
        value = strtrim(strjoin(parts(2:end), ':'));

        currentPath = pathStack{end};
        fullPath = [currentPath, {key}];

        if isempty(value)
            data = set_nested_field(data, fullPath, struct());
            indentStack(end+1) = indent;
            pathStack{end+1} = fullPath;
        else
            data = set_nested_field(data, fullPath, parse_scalar(value));
        end
    end
end

function s = set_nested_field(s, path, value)
    if isempty(path)
        s = value;
        return;
    end
    key = path{1};
    if numel(path) == 1
        s.(key) = value;
    else
        if ~isfield(s, key) || ~isstruct(s.(key))
            s.(key) = struct();
        end
        s.(key) = set_nested_field(s.(key), path(2:end), value);
    end
end

function value = parse_scalar(text)
    if isempty(text)
        value = '';
        return;
    end
    if text(1) == '''' && text(end) == ''''
        value = text(2:end-1);
        return;
    end
    if text(1) == '"' && text(end) == '"'
        value = text(2:end-1);
        return;
    end
    lowerText = lower(text);
    switch lowerText
        case 'true'
            value = true;
            return;
        case 'false'
            value = false;
            return;
    end
    numericValue = str2double(text);
    if ~isnan(numericValue)
        value = numericValue;
    else
        value = text;
    end
end

