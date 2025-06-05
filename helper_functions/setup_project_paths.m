function P = setup_project_paths()
% SETUP_PROJECT_PATHS Validate project root and construct common directories.
%
%   P = SETUP_PROJECT_PATHS() checks that the current working directory
%   contains the expected project folders ('src' and 'data'). It returns a
%   struct with useful paths and ensures base result directories exist.
%   The helper function directory is added to the MATLAB path.
%
%   Fields in P:
%       projectRoot   - absolute path to the project root (pwd)
%       srcPath       - path to the 'src' directory
%       helperFunPath - path to 'src/helper_functions'
%       dataPath      - path to the 'data' directory
%       resultsPath   - path to the 'results' directory
%       modelsPath    - path to the 'models' directory
%       figuresPath   - path to the 'figures' directory
%
%   Example:
%       P = setup_project_paths();
%       modelsP2 = fullfile(P.modelsPath, 'Phase2');
%
%   Date: 2025-05-20

    projectRoot = pwd;
    if ~exist(fullfile(projectRoot, 'src'), 'dir') || ...
       ~exist(fullfile(projectRoot, 'data'), 'dir')
        error(['Project structure not found. Please run scripts from the ' ...
               'project root. Current directory is: %s'], projectRoot);
    end

    P.projectRoot   = projectRoot;
    P.srcPath       = fullfile(projectRoot, 'src');
    P.helperFunPath = fullfile(P.srcPath, 'helper_functions');
    P.dataPath      = fullfile(projectRoot, 'data');
    P.resultsPath   = fullfile(projectRoot, 'results');
    P.modelsPath    = fullfile(projectRoot, 'models');
    P.figuresPath   = fullfile(projectRoot, 'figures');

    baseDirs = {P.resultsPath, P.modelsPath, P.figuresPath};
    for i = 1:numel(baseDirs)
        if ~isfolder(baseDirs{i})
            mkdir(baseDirs{i});
        end
    end

    if exist(P.helperFunPath, 'dir')
        addpath(P.helperFunPath);
    else
        error('Helper functions directory not found: %s', P.helperFunPath);
    end
end
