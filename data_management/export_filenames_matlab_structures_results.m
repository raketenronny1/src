%% export_filenames_matlab_structures_results.m
%
% Lists all files in the main results folder and prints the variables
% contained in each MAT-file. The folder location is resolved relative to
% the project root using `get_project_root`. Set the PROJECT_ROOT
% environment variable to override the default of using the current working
% directory.
%
% Example:
%   projectRoot = get_project_root();
%   folderPath  = fullfile(projectRoot, 'results');
%
projectRoot = get_project_root();
folderPath  = fullfile(projectRoot, 'results');

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
