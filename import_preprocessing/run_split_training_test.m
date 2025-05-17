% ======================================================================
% Script: splitDatasetFTIR.m
% ======================================================================
%   Outputs to .\results\
%     • YYYYMMDD_DatasetSplit_FullLog.txt      (full console log, via diary)
%     • YYYYMMDD_DatasetSplit_Demographics.csv (for Excel/R)
%     • YYYYMMDD_DatasetSplit_Summaries.mat    (summaryTrain & summaryTest)
% ======================================================================

%% 0.  Setup --------------------------------------------------------------
rng('default');                       % or rng(SEED) for reproducible splits

baseDir    = pwd;                     % adjust if desired
resultsDir = fullfile(baseDir,'results');
if ~exist(resultsDir,'dir'), mkdir(resultsDir); end

dateStr  = string(datetime('now','Format','yyyyMMdd'));
logFile  = fullfile(resultsDir,dateStr + "_DatasetSplit_FullLog.txt");
csvFile  = fullfile(resultsDir,dateStr + "_DatasetSplit_Demographics.csv");
matFile  = fullfile(resultsDir,dateStr + "_DatasetSplit_Summaries.mat");

if exist(logFile,'file'), delete(logFile); end
diary(logFile);
fprintf('--- Script started %s ---\n', datestr(now,31));

%% 1.  Fetch dataset ------------------------------------------------------
if ~exist('dataTable','var')
    error('Table "dataTable" not found in workspace.');
end
dataTableRaw = dataTable;
fprintf('dataTable loaded with %d rows\n', height(dataTableRaw));

%% 2.  Normalise variable names ------------------------------------------
varMapping = {
    'ProbeUID', {};
    'Patient_ID',        {'Patient_ID','PatientID','PATIENT_ID'};
    'Fall_ID',           {'Fall_ID','FallID','FALL_ID','Case_ID','CaseID'};
    'WHO_Grade',         {'WHO_Grade','WHO Grade','WHO'};
    'Sex',               {'Sex','Gender'};
    'Age',               {'Age'};
    'methylation_class', {'methylation_class','Methylation_Class','MethylationClass'};
    'CombinedSpectra',   {'CombinedSpectra','Combined Spectra'};
};
for ii = 1:size(varMapping,1)
    stdName = varMapping{ii,1};
    if strcmp(stdName,'ProbeUID'), continue; end
    aliases = varMapping{ii,2};
    idx = find(ismember(lower(dataTableRaw.Properties.VariableNames),lower(aliases)),1);
    if ~isempty(idx) && ~strcmp(dataTableRaw.Properties.VariableNames{idx},stdName)
        fprintf('Renaming "%s" → "%s"\n', ...
             dataTableRaw.Properties.VariableNames{idx}, stdName);
        dataTableRaw = renamevars(dataTableRaw,...
            dataTableRaw.Properties.VariableNames{idx},stdName);
    end
end

%% 3.  Add ProbeUID -------------------------------------------------------
dataTableRaw.ProbeUID = (1:height(dataTableRaw))';
dataTableRaw          = movevars(dataTableRaw,'ProbeUID','Before',1);
initialProbeCount     = height(dataTableRaw);
fprintf('Probe count after UID add: %d\n', initialProbeCount);

%% 4.  Keep only WHO-1 / WHO-3 -------------------------------------------
rawWHO = string(dataTableRaw.WHO_Grade);
rawWHO = regexprep(rawWHO,'^\s*([123])\s*$','WHO-$1','ignorecase');
rawWHO = regexprep(rawWHO,'^\s*I\s*$','WHO-1','ignorecase');
rawWHO = regexprep(rawWHO,'^\s*II\s*$','WHO-2','ignorecase');
rawWHO = regexprep(rawWHO,'^\s*III\s*$','WHO-3','ignorecase');
dataTableRaw.WHO_Grade = categorical(rawWHO);

is13              = ismember(dataTableRaw.WHO_Grade,["WHO-1","WHO-3"]);
dataTableFiltered = dataTableRaw(is13,:);
countAfterWHO13   = height(dataTableFiltered);
fprintf('Probes after WHO-1/3 filter: %d\n', countAfterWHO13);
if countAfterWHO13==0, error('No WHO-1 or WHO-3 probes remain.'); end

%% 5.  Methylation probes → TEST ------------------------------------------
hasMethyl = false(height(dataTableFiltered),1);
if ismember('methylation_class',dataTableFiltered.Properties.VariableNames)
    hasMethyl = ~ismissing(dataTableFiltered.methylation_class);
end
test_Methyl = dataTableFiltered.ProbeUID(hasMethyl);
fprintf('Methylation probes forced to TEST: %d\n', numel(test_Methyl));

%% 6.  Balanced training pool --------------------------------------------
trainPoolMask  = ~hasMethyl;
trainPoolTable = dataTableFiltered(trainPoolMask,:);
ixWHO1 = trainPoolTable.ProbeUID(trainPoolTable.WHO_Grade=='WHO-1');
ixWHO3 = trainPoolTable.ProbeUID(trainPoolTable.WHO_Grade=='WHO-3');
if isempty(ixWHO1) || isempty(ixWHO3)
    finalTrainIDs = [];
    fprintf('Class imbalance: all non-methyl probes → TEST\n');
else
    nPerClass     = min(numel(ixWHO1),numel(ixWHO3));
    finalTrainIDs = [ixWHO1(randperm(numel(ixWHO1),nPerClass)); ...
                     ixWHO3(randperm(numel(ixWHO3),nPerClass))];
    fprintf('Training per class (WHO-1/WHO-3): %d/%d\n',nPerClass,nPerClass);
end

%% 7.  Assemble Train / Test tables --------------------------------------
isTrain        = ismember(dataTableFiltered.ProbeUID,finalTrainIDs);
dataTableTrain = dataTableFiltered(isTrain,:);
dataTableTest  = dataTableFiltered(~isTrain,:);

dataTableTest = [dataTableTest; ...
    dataTableFiltered(ismember(dataTableFiltered.ProbeUID,test_Methyl),:)];
[~,ia] = unique(dataTableTest.ProbeUID,'stable');  % de-dup
dataTableTest = dataTableTest(ia,:);
dataTableTrain = dataTableTrain(~ismember(dataTableTrain.ProbeUID,...
                              dataTableTest.ProbeUID),:);

fprintf('Final TRAIN size: %d  |  TEST size: %d\n', ...
        height(dataTableTrain), height(dataTableTest));

%% 8.  Console demographics ----------------------------------------------
printDemographics(dataTableTrain,'TRAIN');
printDemographics(dataTableTest ,'TEST');

%% 9.  Build CSV demographics --------------------------------------------
summaryRows = {'Category','SubCategory','Group','Metric','Value','Notes'};
addRow      = @(c,sc,g,m,v,n) {c,sc,g,m,v,n};

% — overall dataset counts —
summaryRows = [summaryRows;
               addRow('Dataset','Raw','Overall','ProbeCount',initialProbeCount,'')];
summaryRows = [summaryRows;
               addRow('Dataset','Filtered','Overall','ProbeCount_WHO13',countAfterWHO13,'')];

% — split-logic meta data —
summaryRows = [summaryRows;
               addRow('SplitLogic','TestSet','Overall','MethylationProbes',numel(test_Methyl),'')];

% --- Check for Patient Overlap between Train and Test Sets ---
patientOverlapStatus = 'No'; % Default to 'No'
dupPtsStr = ''; % Default to empty string

if ismember('Patient_ID', dataTableTrain.Properties.VariableNames) && ...
   ismember('Patient_ID', dataTableTest.Properties.VariableNames)

    trainPatientIDs = unique(dataTableTrain.Patient_ID);
    testPatientIDs  = unique(dataTableTest.Patient_ID);

    % Ensure that Patient_IDs are not empty before taking intersect,
    % especially if they could be empty cell arrays or similar.
    if iscell(trainPatientIDs) && isempty(trainPatientIDs)
        trainPatientIDs = {}; % Standardize empty cell
    end
    if iscell(testPatientIDs) && isempty(testPatientIDs)
        testPatientIDs = {}; % Standardize empty cell
    end
    
    % Handle cases where one or both ID sets might be empty after unique()
    % to prevent errors with intersect if types are incompatible (e.g. double and empty cell)
    if (iscell(trainPatientIDs) && isempty(trainPatientIDs)) || (iscell(testPatientIDs) && isempty(testPatientIDs))
        if ~(iscell(trainPatientIDs) && iscell(testPatientIDs)) % If one is cell and other isn't, and one is empty
             if isempty(trainPatientIDs) && ~iscell(testPatientIDs)
                 trainPatientIDs = feval(class(testPatientIDs)); % Make trainPatientIDs empty of same type as test
             elseif isempty(testPatientIDs) && ~iscell(trainPatientIDs)
                 testPatientIDs = feval(class(trainPatientIDs)); % Make testPatientIDs empty of same type as train
             end
        end
    end

    overlappingPatientIDs = intersect(trainPatientIDs, testPatientIDs);

    if ~isempty(overlappingPatientIDs)
        patientOverlapStatus = 'Yes';
        try
            % Step 1: Convert overlapping IDs to a MATLAB string array.
            % This function is quite versatile and handles numeric, categorical,
            % cell arrays of strings, datetime objects, etc.
            stringArray = string(overlappingPatientIDs);
            
            % Step 2: Explicitly handle any <missing> values in the string array.
            % Replace them with a displayable character string like "<missing>".
            stringArray(ismissing(stringArray)) = "<missing>";
            
            % Step 3: Convert the MATLAB string array to a cell array of character vectors.
            % strjoin generally works best and most predictably with cell arrays of char vectors.
            cellArrayOfChars = cellstr(stringArray);
            
            % Step 4: Join the cell array of character vectors into a single string.
            % Ensure it's a row vector if it became a column for strjoin.
            if iscolumn(cellArrayOfChars) && ~isempty(cellArrayOfChars)
                cellArrayOfChars = cellArrayOfChars';
            end
            dupPtsStr = strjoin(cellArrayOfChars, ', ');
            
            % Sanity check: if dupPtsStr is empty but there were IDs, something went wrong.
            if isempty(dupPtsStr) && numel(overlappingPatientIDs) > 0
                 dupPtsStr = '[Overlapping IDs found, but string conversion resulted in empty]';
            end

        catch ME_conversion
            % If any step in the conversion fails, provide a more detailed error.
            dupPtsStr = ['Error during string conversion of overlapping IDs: ' ME_conversion.message];
            % For deeper debugging, you might want to know the class of the problematic variable:
            % disp(['Debug: Class of overlappingPatientIDs causing conversion error: ' class(overlappingPatientIDs)]);
            % And to inspect it in the workspace after script finishes/errors:
            % assignin('base', 'debug_problematic_IDs', overlappingPatientIDs);
        end
        
        fprintf('WARNING: PatientID overlap found between TRAIN and TEST sets: %s\n', dupPtsStr);
    else
        fprintf('No PatientID overlap found between TRAIN and TEST sets.\n');
    end
else
    fprintf('Patient_ID column not found in both TRAIN and TEST tables. Skipping overlap check.\n');
    patientOverlapStatus = 'Unknown - Patient_ID not in both sets';
end
% --- End of Patient Overlap Check ---

% This is your existing line 133 (or around there):
summaryRows = [summaryRows; addRow('SplitLogic','PatientOverlap','Overall','Status',patientOverlapStatus,'')];

% This is your existing conditional block that uses dupPtsStr:
if strcmp(patientOverlapStatus,'Yes') && exist('dupPtsStr','var') && ~isempty(dupPtsStr) && ~startsWith(dupPtsStr, '[Error') && ~startsWith(dupPtsStr, '[Overlapping IDs found, but')
    summaryRows = [summaryRows;
                   addRow('SplitLogic','PatientOverlap','Overall','OverlappingIDs',dupPtsStr,'')];
end
% -----------------------------------------------------------------------

% — detailed demographic rows —
summaryRows = [summaryRows;
               extractAndAddDemographics_for_summary(dataTableTrain,'TrainingSet','Overall'); ...
               extractAndAddDemographics_for_summary(dataTableTrain,'TrainingSet','WHO-1'); ...
               extractAndAddDemographics_for_summary(dataTableTrain,'TrainingSet','WHO-3'); ...
               extractAndAddDemographics_for_summary(dataTableTest,'TestSet','Overall'); ...
               extractAndAddDemographics_for_summary(dataTableTest,'TestSet','WHO-1'); ...
               extractAndAddDemographics_for_summary(dataTableTest,'TestSet','WHO-3')];

% write the CSV
summaryTable = cell2table(summaryRows(2:end,:),...
                          'VariableNames',summaryRows(1,:));
writetable(summaryTable,csvFile);
fprintf('CSV demographics written to: %s\n', csvFile);


%% 10.  Summary structs & MAT --------------------------------------------
summaryTrain = makeSummaryStruct(dataTableTrain);
summaryTest  = makeSummaryStruct(dataTableTest);
save(matFile,'summaryTrain','summaryTest','-v7');
fprintf('MAT summaries written to: %s\n', matFile);

fprintf('--- Script finished %s ---\n', datestr(now,31));
diary off;

% ======================================================================
% Helper functions
% ======================================================================
function printDemographics(tbl,name)
    fprintf('\n%s SET DEMOGRAPHICS (n=%d)\n',name,height(tbl));
    if height(tbl)==0, return; end

    if ismember('WHO_Grade',tbl.Properties.VariableNames)
        cats   = cellstr(categories(tbl.WHO_Grade));
        counts = countcats(tbl.WHO_Grade);
        line   = strjoin(string(arrayfun(@(i) cats{i}+":"+counts(i),1:numel(cats),'UniformOutput',false)),', ');
        fprintf('  WHO distribution : %s\n', line);
    end
    if ismember('Age',tbl.Properties.VariableNames) && isnumeric(tbl.Age)
        fprintf('  Age mean±SD       : %.1f ± %.1f (min %.0f | max %.0f)\n', ...
            mean(tbl.Age,'omitnan'), std(tbl.Age,'omitnan'), ...
            min(tbl.Age,[],'omitnan'), max(tbl.Age,[],'omitnan'));
    end
    if ismember('Sex',tbl.Properties.VariableNames)
        sx   = categorical(tbl.Sex);
        cats = cellstr(categories(sx));
        c    = countcats(sx);
        if any(c)
            line = strjoin(string(arrayfun(@(i) cats{i}+":"+c(i),find(c>0),'UniformOutput',false)),', ');
            fprintf('  Sex distribution  : %s\n', line);
        end
    end
end

function S = makeSummaryStruct(tbl)
    S = struct();
    S.TotalProbes = height(tbl);
    if height(tbl)==0, return; end
    if ismember('WHO_Grade',tbl.Properties.VariableNames)
        S.WHO_Categories   = cellstr(categories(tbl.WHO_Grade));
        S.WHO_Distribution = countcats(tbl.WHO_Grade);
    end
    if ismember('Age',tbl.Properties.VariableNames) && isnumeric(tbl.Age)
        S.Age_Mean = mean(tbl.Age,'omitnan');
        S.Age_Std  = std(tbl.Age,'omitnan');
        S.Age_Min  = min(tbl.Age,[],'omitnan');
        S.Age_Max  = max(tbl.Age,[],'omitnan');
    end
    if ismember('Sex',tbl.Properties.VariableNames)
        sx   = categorical(tbl.Sex);
        S.Sex_Categories = cellstr(categories(sx));
        S.Sex_Counts     = countcats(sx);
    end
end

function rows = extractAndAddDemographics_for_summary(tbl,setName,whoStr)
    rows = {};
    add  = @(grp,met,val,nt) {setName,whoStr,grp,met,val,nt};

    if isempty(tbl)
        if strcmp(whoStr,'Overall')
            rows = [rows; add('Overall','ProbeCount',0,'empty')];
        end
        return
    end

    if ~strcmp(whoStr,'Overall')
        tbl = tbl(tbl.WHO_Grade==whoStr,:);
        if isempty(tbl), return; end
    end

    rows = [rows; add('Overall','ProbeCount',height(tbl),'')];

    if ismember('Age',tbl.Properties.VariableNames) && isnumeric(tbl.Age)
        rows = [rows; add('Age','Mean',mean(tbl.Age,'omitnan'),'years')];
        rows = [rows; add('Age','Std', std(tbl.Age,'omitnan'),'years')];
    end

    if ismember('Sex',tbl.Properties.VariableNames)
        sx   = categorical(tbl.Sex);
        cats = cellstr(categories(sx));
        c    = countcats(sx);
        for k = 1:numel(c)
            if c(k)>0
                rows = [rows; add('Sex',cats{k},c(k),'')];
            end
        end
    end
end