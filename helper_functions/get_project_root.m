%% get_project_root.m
%
% Helper function to determine the base directory of the project.
% The function first checks the environment variable PROJECT_ROOT. If it
% is set and not empty, that value is returned. Otherwise, the current
% working directory (pwd) is returned. This allows scripts to be run from
% anywhere while still locating project files.
%
% Example usage:
%   projectRoot = get_project_root();
%   resultsDir  = fullfile(projectRoot, 'results');
%
% Date: 2025-05-15
%
function rootDir = get_project_root()
    envDir = getenv('PROJECT_ROOT');
    if ~isempty(envDir)
        rootDir = envDir;
    else
        rootDir = pwd;
    end
end
