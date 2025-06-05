%% export_filenames_matlab_structures.m
%
% Lists all files in the Phase3 results folder and prints the variables
% contained in each MAT-file. The project root is determined using the
% helper `get_project_root`, which defaults to the current working
% directory or can be overridden with the PROJECT_ROOT environment
% variable.
%
% Example:
%   projectRoot = get_project_root();
%   folderPath  = fullfile(projectRoot, 'results', 'Phase3');
%
projectRoot = get_project_root();
folderPath  = fullfile(projectRoot, 'results', 'Phase3');

files = dir(folderPath);
files = files(~[files.isdir]);
for i = 1:length(files)
    disp(files(i).name);
end
%% Inspect variables inside MAT-files
files = dir(fullfile(folderPath, '*.mat'));

for i = 1:length(files)
    filePath = fullfile(folderPath, files(i).name);
    fprintf('Variables in %s:\n', files(i).name);
    vars = whos('-file', filePath);
    for v = 1:length(vars)
        fprintf('  %s (%s)\n', vars(v).name, vars(v).class);
    end
    fprintf('\n');
end
