files = dir('C:\Users\Franz\OneDrive\01_Promotion\01 Data\meningioma-ftir-classification\results\Phase3');
files = files(~[files.isdir]);
for i = 1:length(files)
    disp(files(i).name);
end
%%
folderPath = 'C:\Users\Franz\OneDrive\01_Promotion\01 Data\meningioma-ftir-classification\results\Phase3';  % change as needed
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