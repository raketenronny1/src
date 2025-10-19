function dataStruct = load_dataset_split(dataPath, splitName)
%LOAD_DATASET_SPLIT Load train/test dataset tables and flatten spectra.
%   DATA = LOAD_DATASET_SPLIT(DATAPATH, SPLITNAME) loads the specified split
%   ('train' or 'test'), retrieves the wavenumbers, flattens spectra using
%   FLATTEN_SPECTRA_FOR_PCA and returns a struct with fields:
%       .splitName   - 'train' or 'test'
%       .dataTable   - loaded table
%       .wavenumbers - row vector of ROI wavenumbers
%       .X           - spectra matrix (observations x features)
%       .y           - class labels
%       .probeIDs    - probe identifiers per spectrum
%
%   Date: 2025-07-07

    if nargin < 2
        error('load_dataset_split:MissingArguments', 'Both dataPath and splitName are required.');
    end

    splitName = lower(string(splitName));
    if splitName ~= "train" && splitName ~= "test"
        error('load_dataset_split:InvalidSplit', 'splitName must be ''train'' or ''test'' (received %s).', splitName);
    end

    wnFile = fullfile(dataPath, 'wavenumbers.mat');
    if ~isfile(wnFile)
        error('load_dataset_split:MissingWavenumbers', 'Wavenumbers file not found: %s', wnFile);
    end
    wnStruct = load(wnFile, 'wavenumbers_roi');
    if ~isfield(wnStruct, 'wavenumbers_roi')
        error('load_dataset_split:MissingVariable', 'wavenumbers_roi not found in %s', wnFile);
    end
    wavenumbers = wnStruct.wavenumbers_roi;
    if iscolumn(wavenumbers)
        wavenumbers = wavenumbers';
    end

    switch splitName
        case "train"
            tableFile = fullfile(dataPath, 'data_table_train.mat');
            tableVar = 'dataTableTrain';
        case "test"
            tableFile = fullfile(dataPath, 'data_table_test.mat');
            tableVar = 'dataTableTest';
    end

    if ~isfile(tableFile)
        error('load_dataset_split:MissingTable', 'Dataset table not found: %s', tableFile);
    end
    T = load(tableFile, tableVar);
    if ~isfield(T, tableVar)
        error('load_dataset_split:MissingVariable', '%s not found in %s', tableVar, tableFile);
    end
    dataTable = T.(tableVar);

    [X, y, ~, probeIDs] = flatten_spectra_for_pca(dataTable, length(wavenumbers));

    dataStruct = struct();
    dataStruct.splitName = splitName;
    dataStruct.dataTable = dataTable;
    dataStruct.wavenumbers = wavenumbers;
    dataStruct.X = X;
    dataStruct.y = y;
    dataStruct.probeIDs = probeIDs;
end
