%% setup_project_paths.m
%
% Helper to prepare and return common project paths. Optionally specify a
% phase name to create phase specific subfolders.
%
% P = setup_project_paths(projectRoot, phaseName)
%
% Inputs
%   projectRoot - base folder of the repository. If empty or not
%                 provided, get_project_root() is used.
%   phaseName   - optional string like 'Phase2'. If provided, the results,
%                 models and figures paths will include this subfolder.
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
function P = setup_project_paths(projectRoot, phaseName)
    if nargin < 1 || isempty(projectRoot)
        projectRoot = get_project_root();
    end
    if nargin < 2
        phaseName = '';
    end

    % Basic directories
    P.projectRoot    = projectRoot;
    P.srcPath        = fullfile(projectRoot, 'src');
    P.helperFunPath  = fullfile(P.srcPath, 'helper_functions');
    P.pipelinePath   = fullfile(P.srcPath, 'pipelines');
    P.dataPath       = fullfile(projectRoot, 'data');

    P.resultsPath    = fullfile(projectRoot, 'results');
    P.modelsPath     = fullfile(projectRoot, 'models');
    P.figuresPath    = fullfile(projectRoot, 'figures');

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

    % Add helper functions and pipeline classes to the MATLAB path
    if exist(P.helperFunPath, 'dir') && ~contains(path, P.helperFunPath)
        addpath(P.helperFunPath);
    end
    if exist(P.pipelinePath, 'dir') && ~contains(path, P.pipelinePath)
        addpath(P.pipelinePath);
    end
end
