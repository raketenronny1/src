% ------------------------------------------------------------------------------------
% MATLAB SCRIPT: Integrated Workflow to Create Final Probe-Level Table
% ------------------------------------------------------------------------------------
% Creates 'data_probes_final' with specified schema, including raw & processed spectra,
% and formatted categorical variables.
%
% Prerequisites in MATLAB workspace:
%   1. 'allspekTable': User-provided table (~345 rows, one per position).
%      Expected columns: 'RawSpectrum', 'Proben_ID_str', 'Position'.
%   2. 'metadata_patients': User-provided table (~123 rows, from Alle_Proben_Übersicht).
%      Expected columns: 'Proben_ID', 'Diss_ID', 'WHO_Grade', 'Sex', 'Subtyp',
%                        'methylation_class', 'methylation_cluster', 'Fall_ID', 
%                        'Patient', 'Age'.
%   3. 'wavenumbers_roi': Wavenumber vector.
% ------------------------------------------------------------------------------------

%% --- Section 0: Configuration and Prerequisite Checks ---
fprintf('Starting integrated workflow to create data_probes_final...\n');

% --- User-configurable Parameters ---
sg_poly_order = 2; 
sg_frame_len = 11;  

% --- Prerequisite Checks ---
if ~exist('allspekTable', 'var') || ~istable(allspekTable), error('Table "allspekTable" not found.'); end
expected_cols_at = {'RawSpectrum', 'Proben_ID_str', 'Position'};
if any(~ismember(expected_cols_at, allspekTable.Properties.VariableNames)), error('allspekTable missing one or more columns: RawSpectrum, Proben_ID_str, Position.'); end

if ~exist('metadata_patients', 'var') || ~istable(metadata_patients), error('Table "metadata_patients" not found.'); end
expected_cols_mp = {'Proben_ID','Diss_ID','WHO_Grade','Sex','Subtyp','methylation_class','methylation_cluster','Fall_ID','Patient','Age'};
if any(~ismember(expected_cols_mp, metadata_patients.Properties.VariableNames))
    error('metadata_patients missing one or more required columns: %s.', strjoin(setdiff(expected_cols_mp, metadata_patients.Properties.VariableNames),', '));
end

if ~exist('wavenumbers_roi', 'var') || ~isvector(wavenumbers_roi), error('Wavenumber vector "wavenumbers_roi" not found.'); end
if mod(sg_frame_len, 2) == 0 || sg_frame_len <= sg_poly_order, error('Invalid SG params.'); end
if iscolumn(wavenumbers_roi), wavenumbers_roi = wavenumbers_roi'; end
num_wavenumber_points = length(wavenumbers_roi);
if height(allspekTable) > 0 && (~iscell(allspekTable.RawSpectrum) || isempty(allspekTable.RawSpectrum{1}) || size(allspekTable.RawSpectrum{1},2) ~= num_wavenumber_points)
    error('Wavenumber length mismatch with allspekTable.RawSpectrum{1}.');
end

%% --- Part 1: Join allspekTable with metadata_patients to create enriched position table ---
fprintf('\nPart 1: Joining allspekTable with metadata_patients...\n');
allspekTable_j = allspekTable; % Use a copy for join key prep
if iscategorical(allspekTable_j.Proben_ID_str), allspekTable_j.Proben_ID_join_key = string(strtrim(cellstr(allspekTable_j.Proben_ID_str)));
else, allspekTable_j.Proben_ID_join_key = string(strtrim(cellstr(allspekTable_j.Proben_ID_str))); end

metadata_patients_j = metadata_patients;
if isnumeric(metadata_patients_j.Proben_ID), metadata_patients_j.Proben_ID_join_key = string(metadata_patients_j.Proben_ID);
else, metadata_patients_j.Proben_ID_join_key = string(strtrim(cellstr(metadata_patients_j.Proben_ID))); end

try
    data_all_positions_temp = innerjoin(allspekTable_j, metadata_patients_j, ...
        'LeftKeys', 'Proben_ID_join_key', 'RightKeys', 'Proben_ID_join_key', ...
        'RightVariables', setdiff(metadata_patients_j.Properties.VariableNames, {'Proben_ID_join_key', 'Proben_ID'}));
    if ismember('Proben_ID_join_key', data_all_positions_temp.Properties.VariableNames)
         data_all_positions_temp.Proben_ID_join_key = []; end
    fprintf('Join complete. data_all_positions_temp created with %d rows.\n', height(data_all_positions_temp));
    if height(data_all_positions_temp) == 0, error('No matches found in join.'); end
catch ME, error('Failed to join tables: %s', ME.message); end

%% --- Part 2: Apply Sequential Spectral Preprocessing to position-level data ---
fprintf('\nPart 2: Applying sequential preprocessing to data_all_positions_temp.RawSpectrum...\n');
numBlocks_p2 = height(data_all_positions_temp);
ProcessedSpectra_ForEachPosition_cell = cell(numBlocks_p2, 1);

for i = 1:numBlocks_p2
    raw_block = data_all_positions_temp.RawSpectrum{i};
    if isempty(raw_block) || ~ismatrix(raw_block) || size(raw_block,2) ~= num_wavenumber_points
        warning('Row %d (Pos %s, Diss_ID %s): RawSpectrum invalid. Preprocessing skipped for this block.', i, data_all_positions_temp.Position{i}, data_all_positions_temp.Diss_ID{i});
        ProcessedSpectra_ForEachPosition_cell{i} = raw_block; continue;
    end
    % 2a. Smoothing
    smooth_b=zeros(size(raw_block)); for j=1:size(raw_block,1),s=raw_block(j,:);if any(isnan(s))||length(s)<sg_frame_len,smooth_b(j,:)=s;else,try,smooth_b(j,:)=sgolayfilt(s,sg_poly_order,sg_frame_len);catch,smooth_b(j,:)=s;end;end;end
    % 2b. SNV
    snv_b=zeros(size(smooth_b)); for j=1:size(smooth_b,1),s=smooth_b(j,:);if any(isnan(s))||all(s==0,'all'),snv_b(j,:)=s;else,m=mean(s,'omitnan');sd=std(s,0,1,'omitnan');if sd<eps,snv_b(j,:)=s;else,snv_b(j,:)=(s-m)/sd;end;end;end
    % 2c. L2-Norm
    l2_b=zeros(size(snv_b)); for j=1:size(snv_b,1),s=snv_b(j,:);if any(isnan(s))||all(s==0,'all'),l2_b(j,:)=s;else,n=norm(s,2);if abs(n)<eps,l2_b(j,:)=s;else,l2_b(j,:)=s/n;end;end;end
    ProcessedSpectra_ForEachPosition_cell{i} = l2_b;
end
data_all_positions_temp.FinalProcessedSpectrum_pos = ProcessedSpectra_ForEachPosition_cell; % Spectra per position, processed
disp('Sequential spectral preprocessing complete for all positions.');

%% --- Part 3: Aggregate to data_probes_final and Format Variables per Schema ---
fprintf('\nPart 3: Aggregating to data_probes_final and formatting variables...\n');
unique_diss_ids_final = unique(data_all_positions_temp.Diss_ID);
num_unique_probes_final = length(unique_diss_ids_final);

% Initialize cell array for table construction
% Target schema: Diss_ID(cellstr), Patient_ID(string), Fall_ID(double), WHO_Grade(cat), Sex(cat), Age(double), Subtyp(cat),
% methylation_class(cat), methylation_cluster(cat), NumPositions(double),
% PositionSpectra(cell), NumTotalSpectra(double), CombinedRawSpectra(cell), CombinedSpectra(cell), MeanSpectrum(cell)
num_final_cols = 15; % Added CombinedRawSpectra
probe_data_final_cell = cell(num_unique_probes_final, num_final_cols);

for i = 1:num_unique_probes_final
    current_diss_id = unique_diss_ids_final{i};
    idx_this_probe = strcmp(data_all_positions_temp.Diss_ID, current_diss_id);
    table_this_probe_positions = data_all_positions_temp(idx_this_probe, :);
    num_positions_for_probe = height(table_this_probe_positions);

    % Initialize for aggregation
    combined_raw_spectra_agg = [];
    combined_processed_spectra_agg = [];
    position_spectra_detail_agg = cell(num_positions_for_probe, 2); % {'PosName', ProcessedSpectraMatrixForPos}

    for k_pos = 1:num_positions_for_probe
        current_pos_data_loop = table_this_probe_positions(k_pos, :);
        
        raw_spec_block = current_pos_data_loop.RawSpectrum{1};
        if ~isempty(raw_spec_block) && ismatrix(raw_spec_block)
            combined_raw_spectra_agg = [combined_raw_spectra_agg; raw_spec_block];
        end
        
        processed_spec_block = current_pos_data_loop.FinalProcessedSpectrum_pos{1};
        if ~isempty(processed_spec_block) && ismatrix(processed_spec_block)
            combined_processed_spectra_agg = [combined_processed_spectra_agg; processed_spec_block];
            position_spectra_detail_agg{k_pos, 1} = current_pos_data_loop.Position{1};
            position_spectra_detail_agg{k_pos, 2} = processed_spec_block; % Store the processed block for this position
        else % Handle case where processed block might be empty if raw was invalid
            position_spectra_detail_agg{k_pos, 1} = current_pos_data_loop.Position{1};
            position_spectra_detail_agg{k_pos, 2} = cell(0,num_wavenumber_points); % Empty matrix with correct columns
        end
    end
    
    meta_src_final = table_this_probe_positions(1, :); % Metadata from first position
    
    % WHO_Grade string prep
    raw_who_val = meta_src_final.WHO_Grade; who_grade_str_final = ''; val_check_who='';
    % ... (Full WHO_Grade string conversion logic from response #57) ...
    if isnumeric(raw_who_val),val_check_who=num2str(raw_who_val); elseif isstring(raw_who_val)||ischar(raw_who_val),val_check_who=char(raw_who_val); elseif iscell(raw_who_val)&&~isempty(raw_who_val),val_check_who=char(raw_who_val{1}); elseif iscategorical(raw_who_val), if isundefined(raw_who_val),val_check_who='';else,val_check_who=char(raw_who_val);end;end; val_check_who=regexprep(val_check_who,'\s*\(\d+\)$',''); val_num_who=str2double(val_check_who); if ~isnan(val_num_who), if val_num_who==1,who_grade_str_final='WHO-1'; elseif val_num_who==2,who_grade_str_final='WHO-2'; elseif val_num_who==3,who_grade_str_final='WHO-3';end;else, if strcmpi(strtrim(val_check_who),'WHO-1'),who_grade_str_final='WHO-1'; elseif strcmpi(strtrim(val_check_who),'WHO-2'),who_grade_str_final='WHO-2'; elseif strcmpi(strtrim(val_check_who),'WHO-3'),who_grade_str_final='WHO-3';end;end; if isempty(who_grade_str_final) && ~isempty(val_check_who) && ~any(strcmpi(val_check_who, {'<missing>','NaN',''})), warning('Unmapped WHO_Grade for Diss_ID %s: original "%s"', current_diss_id, val_check_who); end

    % Sex string prep
    raw_sex_val = meta_src_final.Sex; sex_str_final = ''; val_check_sex='';
    % ... (Full Sex string conversion logic from response #57) ...
    if ischar(raw_sex_val),val_check_sex=strtrim(lower(raw_sex_val)); elseif iscellstr(raw_sex_val)&&~isempty(raw_sex_val),val_check_sex=strtrim(lower(raw_sex_val{1})); elseif isstring(raw_sex_val)&&strlength(raw_sex_val)>0,val_check_sex=strtrim(lower(char(raw_sex_val))); elseif iscategorical(raw_sex_val), if isundefined(raw_sex_val),val_check_sex='';else,val_check_sex=strtrim(lower(char(raw_sex_val)));end;end; if strcmp(val_check_sex,'w')||strcmp(val_check_sex,'female'),sex_str_final='Female'; elseif strcmp(val_check_sex,'m')||strcmp(val_check_sex,'male'),sex_str_final='Male';end; if isempty(sex_str_final)&&~isempty(val_check_sex) && ~any(strcmpi(val_check_sex, {'<missing>','NaN',''})), warning('Unmapped Sex for Diss_ID %s: original "%s"', current_diss_id, val_check_sex); end

    % Subtyp string prep
    raw_sub_val = meta_src_final.Subtyp; subtyp_str_final = ''; val_check_sub='';
    map_sub=containers.Map({'fibromatös','meningothelial','transitional','klarzellig','chordoid','anaplastisch','atypisch','psammomatös'},{'fibro','meningo','trans','clear','chord','anap','atyp','psamm'});
    % ... (Full Subtyp string conversion logic from response #57, using map_sub) ...
    if ischar(raw_sub_val),val_check_sub=strtrim(raw_sub_val); elseif iscellstr(raw_sub_val)&&~isempty(raw_sub_val),val_check_sub=strtrim(raw_sub_val{1}); elseif isstring(raw_sub_val)&&strlength(raw_sub_val)>0,val_check_sub=strtrim(char(raw_sub_val)); elseif iscategorical(raw_sub_val), if isundefined(raw_sub_val),val_check_sub='';else,val_check_sub=strtrim(char(raw_sub_val));end;end; if isKey(map_sub,val_check_sub),subtyp_str_final=map_sub(val_check_sub); else, if ~isempty(val_check_sub) && ~any(strcmpi(val_check_sub, {'<missing>','NaN',''})),warning('Unmapped Subtyp for Diss_ID %s: original "%s"',current_diss_id,val_check_sub); subtyp_str_final=val_check_sub; else subtyp_str_final='';end;end

    patient_id_val = string(meta_src_final.Patient); % Ensure string type
    fall_id_raw_val = meta_src_final.Fall_ID;
    % Attempt to convert Fall_ID to double, handle non-numeric gracefully
    if ischar(fall_id_raw_val) || isstring(fall_id_raw_val) || (iscell(fall_id_raw_val) && ~isempty(fall_id_raw_val))
        if iscell(fall_id_raw_val), fall_id_raw_val = fall_id_raw_val{1}; end
        fall_id_double = str2double(fall_id_raw_val); % Converts to NaN if not purely numeric
        if isnan(fall_id_double) && ~isempty(strtrim(char(fall_id_raw_val))) && ~strcmpi(strtrim(char(fall_id_raw_val)),'nan')
             warning('Fall_ID "%s" for Diss_ID %s is not purely numeric, converted to NaN.', char(fall_id_raw_val), current_diss_id);
        end
    elseif isnumeric(fall_id_raw_val)
        fall_id_double = double(fall_id_raw_val);
    else
        fall_id_double = NaN; % Default for other types or empty
        if ~isempty(fall_id_raw_val) && ~ismissing(fall_id_raw_val) % if not truly empty or missing
            warning('Fall_ID for Diss_ID %s has unexpected type, converted to NaN.', current_diss_id);
        end
    end
    
    age_val = meta_src_final.Age; % Should be double
    mc_raw = meta_src_final.methylation_class;
    mcl_raw = meta_src_final.methylation_cluster;
    
    num_total_spectra_final = size(combined_processed_spectra_agg, 1);
    mean_processed_spectrum_final = [];
    if num_total_spectra_final > 0
        mean_processed_spectrum_final = mean(combined_processed_spectra_agg, 1, 'omitnan');
    else
        mean_processed_spectrum_final = NaN(1, num_wavenumber_points);
    end
    
    probe_data_final_cell(i,:) = { ...
        current_diss_id, patient_id_val, fall_id_double, who_grade_str_final, sex_str_final, ...
        age_val, subtyp_str_final, mc_raw, mcl_raw, num_positions_for_probe, ...
        {position_spectra_detail_agg}, num_total_spectra_final, {combined_raw_spectra_agg}, ...
        {combined_processed_spectra_agg}, {mean_processed_spectrum_final} ...
    };
end

data_probes_final = cell2table(probe_data_final_cell, 'VariableNames', { ...
    'Diss_ID', 'Patient_ID', 'Fall_ID', 'WHO_Grade_str', 'Sex_str', 'Age', 'Subtyp_str', ...
    'methylation_class_raw', 'methylation_cluster_raw', 'NumPositions', ...
    'PositionSpectra', 'NumTotalSpectra', 'CombinedRawSpectra', ...
    'CombinedSpectra', 'MeanSpectrum'}); % Matched user's schema names

% --- Final Data Type Conversions for data_probes_final ---
fprintf('\nFinalizing data types in data_probes_final...\n');

% Diss_ID: Ensure it's cell array of char vectors as per schema (cell2table might make it string if all are non-empty)
if isstring(data_probes_final.Diss_ID)
    data_probes_final.Diss_ID = cellstr(data_probes_final.Diss_ID);
end
fprintf('  Diss_ID type: %s\n', class(data_probes_final.Diss_ID));

% Patient_ID: string (already set during creation)
fprintf('  Patient_ID type: %s\n', class(data_probes_final.Patient_ID));

% Fall_ID: double (already set during creation, with NaNs for non-convertibles)
fprintf('  Fall_ID type: %s (NaNs if original was non-numeric text)\n', class(data_probes_final.Fall_ID));

% WHO_Grade: categorical
defined_who_cats = {'WHO-1', 'WHO-2', 'WHO-3'};
data_probes_final.WHO_Grade = categorical(data_probes_final.WHO_Grade_str, defined_who_cats, 'Protected', true);
fprintf('  WHO_Grade type: %s, Categories: %s\n', class(data_probes_final.WHO_Grade), strjoin(categories(data_probes_final.WHO_Grade(find(~isundefined(data_probes_final.WHO_Grade),1,'first'))),', '));

% Sex: categorical
defined_sex_cats = {'Male', 'Female'};
data_probes_final.Sex = categorical(data_probes_final.Sex_str, defined_sex_cats, 'Protected', true);
fprintf('  Sex type: %s, Categories: %s\n', class(data_probes_final.Sex), strjoin(categories(data_probes_final.Sex(find(~isundefined(data_probes_final.Sex),1,'first'))),', '));

% Subtyp: categorical (target 8 categories)
defined_subtyp_cats_target = {'fibro','meningo','trans','atyp','anap','clear','chord','psamm'};
all_unique_subtyps_in_data = unique(data_probes_final.Subtyp_str(~cellfun('isempty',data_probes_final.Subtyp_str)));
final_subtyp_category_list = union(defined_subtyp_cats_target, all_unique_subtyps_in_data', 'stable');
data_probes_final.Subtyp = categorical(data_probes_final.Subtyp_str, final_subtyp_category_list, 'Protected', true);
fprintf('  Subtyp type: %s, Categories (%d): %s\n', class(data_probes_final.Subtyp), length(categories(data_probes_final.Subtyp(find(~isundefined(data_probes_final.Subtyp),1,'first')))), strjoin(categories(data_probes_final.Subtyp(find(~isundefined(data_probes_final.Subtyp),1,'first'))),', '));

% methylation_class: categorical (target 4 categories - determined by data)
missing_indicators = {'' '-' 'NA' 'N/A'};
if ismember('methylation_class_raw', data_probes_final.Properties.VariableNames)
    mc_raw_col = data_probes_final.methylation_class_raw; mc_str_col = cell(size(mc_raw_col));
    for k=1:length(mc_raw_col),val=mc_raw_col(k);if ismissing(val),mc_str_col{k}='';elseif isnumeric(val),mc_str_col{k}=num2str(val);else,mc_str_col{k}=char(val);end;end
    data_probes_final.methylation_class = categorical(standardizeMissing(mc_str_col, missing_indicators));
    fprintf('  methylation_class type: %s, Categories (%d): %s\n', class(data_probes_final.methylation_class), length(categories(data_probes_final.methylation_class(find(~isundefined(data_probes_final.methylation_class),1,'first')))), strjoin(categories(data_probes_final.methylation_class(find(~isundefined(data_probes_final.methylation_class),1,'first'))),', '));
end

% methylation_cluster: categorical (target 6 categories - determined by data)
if ismember('methylation_cluster_raw', data_probes_final.Properties.VariableNames)
    mcl_raw_col = data_probes_final.methylation_cluster_raw; mcl_str_col = cell(size(mcl_raw_col));
    for k=1:length(mcl_raw_col),val=mcl_raw_col(k);if ismissing(val),mcl_str_col{k}='';elseif isnumeric(val),mcl_str_col{k}=num2str(val);else,mcl_str_col{k}=char(val);end;end
    data_probes_final.methylation_cluster = categorical(standardizeMissing(mcl_str_col, missing_indicators));
    fprintf('  methylation_cluster type: %s, Categories (%d): %s\n', class(data_probes_final.methylation_cluster), length(categories(data_probes_final.methylation_cluster(find(~isundefined(data_probes_final.methylation_cluster),1,'first')))), strjoin(categories(data_probes_final.methylation_cluster(find(~isundefined(data_probes_final.methylation_cluster),1,'first'))),', '));
end

% Remove helper string/raw columns
vars_to_remove = {'WHO_Grade_str', 'Sex_str', 'Subtyp_str', 'methylation_class_raw', 'methylation_cluster_raw'};
vars_exist_to_remove = intersect(vars_to_remove, data_probes_final.Properties.VariableNames);
if ~isempty(vars_exist_to_remove)
    data_probes_final = removevars(data_probes_final, vars_exist_to_remove);
end

fprintf('\n--- Integrated Workflow Fully Complete. "data_probes_final" table created and formatted. ---\n');
disp('Final schema of data_probes_final (first few rows, selected columns):');
final_schema_cols = {'Diss_ID','Patient_ID','Fall_ID','WHO_Grade','Sex','Age','Subtyp','methylation_class','methylation_cluster','NumPositions','PositionSpectra','NumTotalSpectra','CombinedRawSpectra','CombinedSpectra','MeanSpectrum'};
disp(head(data_probes_final(:, intersect(final_schema_cols, data_probes_final.Properties.VariableNames, 'stable'))));

% Verify column types (examples)
if height(data_probes_final)>0
    fprintf('\nVerification of selected column types in data_probes_final:\n');
    fprintf('  Diss_ID: %s\n', class(data_probes_final.Diss_ID));
    fprintf('  Patient_ID: %s\n', class(data_probes_final.Patient_ID));
    fprintf('  Fall_ID: %s\n', class(data_probes_final.Fall_ID));
    fprintf('  WHO_Grade: %s\n', class(data_probes_final.WHO_Grade));
    fprintf('  PositionSpectra type of content: %s (expected cell with {PosName, SpectraMatrix})\n', class(data_probes_final.PositionSpectra{1}));
    fprintf('  CombinedRawSpectra type of content: %s (expected cell with spectra matrix)\n', class(data_probes_final.CombinedRawSpectra{1}));
    fprintf('  CombinedSpectra type of content: %s (expected cell with spectra matrix)\n', class(data_probes_final.CombinedSpectra{1}));
end