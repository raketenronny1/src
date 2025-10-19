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
    if ~isempty(envDir) && isfolder(envDir)
        rootDir = envDir;
        return;
    end

    % Try to discover the repository root by walking upwards from the
    % current working directory and from this helper's location. We look for
    % a directory that appears to be the project base (contains a .git
    % folder or the expected "src" directory).
    searchStarts = {pwd, fileparts(mfilename('fullpath'))};
    searchStarts = searchStarts(~cellfun(@isempty, searchStarts));
    [~, uniqueIdx] = unique(searchStarts, 'stable');
    searchStarts = searchStarts(uniqueIdx);
    for i = 1:numel(searchStarts)
        candidate = locate_repo_root(searchStarts{i});
        if ~isempty(candidate)
            rootDir = candidate;
            return;
        end
    end

    % Fallback – if no marker was found just return the current directory.
    rootDir = pwd;
end

function root = locate_repo_root(startDir)
    root = '';
    if ~isfolder(startDir)
        return;
    end

    currentDir = startDir;
    while true
        % Check if this is the repository root
        % The root should have a .git folder, OR it should have both:
        % - a 'src' subdirectory AND
        % - either a 'results' directory or 'README.md' file
        hasGit = isfolder(fullfile(currentDir, '.git'));
        hasSrc = isfolder(fullfile(currentDir, 'src'));
        hasResults = isfolder(fullfile(currentDir, 'results'));
        hasReadme = isfile(fullfile(currentDir, 'README.md'));
        
        % IMPORTANT: Make sure we're not IN the src directory itself
        % by checking if the parent directory exists and has what we need
        [~, dirName] = fileparts(currentDir);
        isInSrcDir = strcmp(dirName, 'src');
        
        if hasGit || (hasSrc && (hasResults || hasReadme) && ~isInSrcDir)
            root = currentDir;
            return;
        end

        parentDir = fileparts(currentDir);
        if strcmp(parentDir, currentDir) || isempty(parentDir)
            return;
        end
        currentDir = parentDir;
    end
end