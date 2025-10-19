%% setup_project_paths.m
%
% Helper to prepare and return common project paths. Optionally specify a
% phase name to create phase specific subfolders.
%
% P = setup_project_paths(projectRoot, phaseName, cfg)
%
% Inputs
%   projectRoot - base folder of the repository. If empty or not
%                 provided, get_project_root() is used.
%   phaseName   - optional string like 'Phase2'. If provided, the results,
%                 models and figures paths will include this subfolder.
%   cfg         - optional configuration struct. When provided, the helper
%                 respects cfg.dataDir, cfg.resultsDir, cfg.modelsDir and
%                 cfg.figuresDir values (or cfg.paths.<field> equivalents).
%
% The returned struct P contains fields:
%   projectRoot, srcPath, helperFunPath, dataPath,
%   resultsPath, modelsPath, figuresPath
%
% Directories are created if they do not exist. The helper functions path is
% added to the MATLAB search path if needed.
%
% Date: 2025-06-05
%
% Example:
%   P = setup_project_paths([], 'Phase2');
%   dataPath = P.dataPath;
%
function P = setup_project_paths(projectRoot, phaseName, cfg)
    if nargin < 1 || isempty(projectRoot)
        projectRoot = get_project_root();
    end
    if nargin < 2
        phaseName = '';
    end
    if nargin < 3 || ~isstruct(cfg)
        cfg = struct();
    end

    % Basic directories
    P.projectRoot    = projectRoot;
    P.srcPath        = fullfile(projectRoot, 'src');
    P.helperFunPath  = fullfile(P.srcPath, 'helper_functions');

    dataDirName    = get_cfg_dir(cfg, 'dataDir', 'data');
    resultsDirName = get_cfg_dir(cfg, 'resultsDir', 'results');
    modelsDirName  = get_cfg_dir(cfg, 'modelsDir', 'models');
    figuresDirName = get_cfg_dir(cfg, 'figuresDir', 'figures');

    P.dataPath    = resolve_dir(projectRoot, dataDirName);
    P.resultsPath = resolve_dir(projectRoot, resultsDirName);
    P.modelsPath  = resolve_dir(projectRoot, modelsDirName);
    P.figuresPath = resolve_dir(projectRoot, figuresDirName);

    if ~isempty(phaseName)
        P.resultsPath = fullfile(P.resultsPath, phaseName);
        P.modelsPath  = fullfile(P.modelsPath, phaseName);
        P.figuresPath = fullfile(P.figuresPath, phaseName);
    end

    % Ensure directories exist
    dirFields = {'resultsPath','modelsPath','figuresPath'};
    for i=1:numel(dirFields)
        d = P.(dirFields{i});
        if ~isempty(d) && ~isfolder(d)
            mkdir(d);
        end
    end

    % Add helper functions to path
    if exist(P.helperFunPath, 'dir') && ~contains(path, P.helperFunPath)
        addpath(P.helperFunPath);
    end
end

function dirName = get_cfg_dir(cfg, fieldName, defaultValue)
    dirName = defaultValue;
    if isfield(cfg, fieldName) && ~isempty(cfg.(fieldName))
        dirName = cfg.(fieldName);
        return;
    end
    if isfield(cfg, 'paths') && isstruct(cfg.paths) && ...
            isfield(cfg.paths, fieldName) && ~isempty(cfg.paths.(fieldName))
        dirName = cfg.paths.(fieldName);
    end
end

function pathOut = resolve_dir(projectRoot, dirName)
    if isstring(dirName)
        dirName = char(dirName);
    end
    if isempty(dirName)
        pathOut = projectRoot;
        return;
    end
    if dirName(1) == '~'
        userDir = char(java.lang.System.getProperty('user.home'));
        pathOut = fullfile(userDir, dirName(2:end));
    elseif is_absolute_path(dirName)
        pathOut = dirName;
    else
        pathOut = fullfile(projectRoot, dirName);
    end
end

function tf = is_absolute_path(pathStr)
    if isempty(pathStr)
        tf = false;
        return;
    end
    tf = startsWith(pathStr, filesep) || ...
        ~isempty(regexp(pathStr, '^[A-Za-z]:', 'once'));
end
