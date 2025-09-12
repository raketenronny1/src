function make_paper_figs_and_tables
% Create paper-ready tables and figures from Phase 2/3 results.
% Outputs:
%   tables/Table1_CV_Comparison.csv / .tex
%   tables/Table2_Probe_Comparison.csv / .tex
%   tables/TableS1_HyperparamStability.csv / .tex
%   figures/Fig1_AUC_withErrorBars.pdf (vector)
%   figures/FigS1_AllMetrics_withErrorBars.pdf (vector)

clc; close all; rng('default')
set(groot,'defaultAxesFontName','Helvetica','defaultAxesFontSize',9);
set(groot,'defaultTextInterpreter','none'); % keep labels literal

%% === Paths (edit if needed) ===
analysisDate = '20250912'; % <-- EDIT THIS DATE FOR NEW ANALYSES
phase2File = sprintf('%s_Phase2_AllPipelineResults.mat', analysisDate);
phase3File = sprintf('%s_Phase3_ComparisonResults.mat', analysisDate);
outTabDir  = fullfile(pwd,'tables');   if ~exist(outTabDir,'dir'), mkdir(outTabDir); end
outFigDir  = fullfile(pwd,'figures');  if ~exist(outFigDir,'dir'), mkdir(outFigDir); end

%% === Load Phase 2 ===
if ~exist(phase2File, 'file')
    error('Phase 2 file not found: %s', phase2File);
end
S2 = load(phase2File);

% Check if required fields exist
required_fields = {'metricNames', 'numOuterFolds', 'numInnerFolds', 'resultsPerPipeline'};
for i = 1:length(required_fields)
    if ~isfield(S2, required_fields{i})
        error('Required field missing in Phase 2 data: %s', required_fields{i});
    end
end

metricNames = string(S2.metricNames(:)');
Kouter = S2.numOuterFolds; Kinner = S2.numInnerFolds;

% Collect mean and per-fold metrics for each pipeline
P = numel(S2.resultsPerPipeline);
pipeNames = strings(P,1);
MEAN = nan(P, numel(metricNames));
SD   = nan(P, numel(metricNames));
RAW  = cell(P,1);

% Debug: Print available fields in first pipeline
if P > 0
    fprintf('Available fields in resultsPerPipeline(1):\n');
    disp(fieldnames(S2.resultsPerPipeline(1)));
    if isfield(S2.resultsPerPipeline(1), 'pipelineConfig')
        fprintf('Available fields in pipelineConfig:\n');
        disp(fieldnames(S2.resultsPerPipeline(1).pipelineConfig));
    end
end

for p = 1:P
    r = S2.resultsPerPipeline(p);
    
    % Extract pipeline name (try multiple possible field names)
    if isfield(r, 'pipelineConfig') && isfield(r.pipelineConfig, 'name')
        pipeNames(p) = string(r.pipelineConfig.name);
    elseif isfield(r, 'name')
        pipeNames(p) = string(r.name);
    elseif isfield(r, 'pipelineName')
        pipeNames(p) = string(r.pipelineName);
    elseif isfield(r, 'config') && isfield(r.config, 'name')
        pipeNames(p) = string(r.config.name);
    else
        warning('Pipeline %d missing name, using default', p);
        pipeNames(p) = sprintf("Pipeline_%d", p);
    end
    
    % Extract metrics data (try multiple possible field names)
    raw_data = [];
    mean_data = [];
    
    % Try different possible field names for raw metrics
    raw_field_names = {'outerFoldMetrics_raw', 'metrics_raw', 'rawMetrics', 'foldMetrics', 'outerFoldMetrics'};
    for fn = raw_field_names
        if isfield(r, fn{1})
            raw_data = r.(fn{1});
            break;
        end
    end
    
    % Try different possible field names for mean metrics
    mean_field_names = {'outerFoldMetrics_mean', 'metrics_mean', 'meanMetrics', 'avgMetrics', 'outerFoldMetrics_avg'};
    for fn = mean_field_names
        if isfield(r, fn{1})
            mean_data = r.(fn{1});
            break;
        end
    end
    
    % If we didn't find mean data but have raw data, compute mean
    if isempty(mean_data) && ~isempty(raw_data)
        mean_data = mean(raw_data, 1, 'omitnan');
        fprintf('Computed mean for pipeline %d from raw data\n', p);
    end
    
    % If we still don't have data, try to find any numeric matrix
    if isempty(raw_data)
        field_names = fieldnames(r);
        for fn = field_names'
            field_data = r.(fn{1});
            if isnumeric(field_data) && ismatrix(field_data) && size(field_data, 2) == numel(metricNames)
                raw_data = field_data;
                fprintf('Using field "%s" as raw metrics for pipeline %d\n', fn{1}, p);
                break;
            end
        end
    end
    
    if isempty(raw_data)
        error('Pipeline %d: Cannot find metrics data. Available fields: %s', p, strjoin(fieldnames(r), ', '));
    end
    
    % Ensure data dimensions are correct
    if size(raw_data, 2) ~= numel(metricNames)
        warning('Pipeline %d: Metric dimensions mismatch. Expected %d metrics, got %d', p, numel(metricNames), size(raw_data, 2));
        % Try to adjust if possible
        if size(raw_data, 2) < numel(metricNames)
            % Pad with NaNs
            raw_data = [raw_data, nan(size(raw_data, 1), numel(metricNames) - size(raw_data, 2))];
        else
            % Truncate
            raw_data = raw_data(:, 1:numel(metricNames));
        end
    end
    
    RAW{p} = raw_data;
    if ~isempty(mean_data)
        MEAN(p,:) = mean_data(:)';
    else
        MEAN(p,:) = mean(raw_data, 1, 'omitnan');
    end
    SD(p,:) = std(RAW{p}, 0, 1, 'omitnan');       % sample SD across outer folds
end

% === Table 1: Nested CV comparison (mean ± SD) ===
fmt = @(m,s) arrayfun(@(mi,si) sprintf('%.3f ± %.3f', mi, si), m, s, 'UniformOutput', false);
T1 = table(pipeNames, 'VariableNames', {'Pipeline'});
for m = 1:numel(metricNames)
    T1.(metricNames(m)) = fmt(MEAN(:,m), SD(:,m));
end

% Bold best per metric (higher is better) using LaTeX mark-up
T1_tex = T1;
for m = 1:numel(metricNames)
    [~,ix] = max(MEAN(:,m));
    current_col = T1_tex{:, m+1}; % Get the entire column
    current_col{ix} = ['\textbf{', current_col{ix}, '}'];
    T1_tex{:, m+1} = current_col; % Assign back to table
end

writetable(T1, fullfile(outTabDir,'Table1_CV_Comparison.csv'));
table2latex(T1_tex, fullfile(outTabDir,'Table1_CV_Comparison.tex'), ...
    'Caption', sprintf('Nested %dx%d CV performance (mean ± SD).', Kouter, Kinner), ...
    'Label', 'tab:cv_comparison');

% === Fig 1: AUC with error bars across pipelines ===
aucIdx = find(strcmp(metricNames, "AUC"), 1);
if ~isempty(aucIdx)
    fig = figure('Units','centimeters','Position',[2 2 9 6]);
    hold on
    x = 1:P;
    hB = bar(x, MEAN(:,aucIdx));
    er = errorbar(x, MEAN(:,aucIdx), SD(:,aucIdx), 'LineStyle','none');
    er.Color = [0 0 0];
    xticks(x); xticklabels(pipeNames); xtickangle(20)
    ylabel('AUC'); ylim([0 1]); grid on
    title('Cross-validated AUC by pipeline')
    exportgraphics(fig, fullfile(outFigDir,'Fig1_AUC_withErrorBars.pdf'), 'ContentType','vector');
    close(fig);
end

% === Optional: All metrics with error bars (supplement) ===
fig = figure('Units','centimeters','Position',[2 2 14 9]);
tiledlayout('flow')
for m = 1:numel(metricNames)
    nexttile
    bar(MEAN(:,m)); hold on
    errorbar(1:P, MEAN(:,m), SD(:,m), 'LineStyle','none', 'Color', 'k');
    title(char(metricNames(m))); ylim([0 1]); grid on
    xticks(1:P); xticklabels(pipeNames); xtickangle(30)
end
exportgraphics(fig, fullfile(outFigDir,'FigS1_AllMetrics_withErrorBars.pdf'), 'ContentType','vector');
close(fig);

% === Table S1: Hyperparameter stability across folds ===
HP = struct();  % struct of arrays
for p = 1:P
    if ~isfield(S2.resultsPerPipeline(p), 'outerFoldBestHyperparams')
        warning('Pipeline %d missing hyperparameter data', p);
        continue;
    end
    
    hpCells = S2.resultsPerPipeline(p).outerFoldBestHyperparams;
    if isempty(hpCells)
        continue;
    end
    
    names = fieldnames(hpCells(1));
    for k = 1:numel(names)
        v = arrayfun(@(h) h.(names{k}), hpCells);
        key = names{k};
        if ~isfield(HP, key), HP.(key) = table(); end
        
        % Count frequencies per pipeline
        cats = unique(v);
        cnts = arrayfun(@(c) sum(v==c), cats);
        
        T = table(repmat(string(pipeNames(p)), length(cats), 1), ...
                  categorical(cats(:)), cnts(:), ...
                  'VariableNames', {'Pipeline', 'Value', 'Count'});
        HP.(key) = [HP.(key); T];
    end
end

% Write all HP tables
hpFields = fieldnames(HP);
for i = 1:numel(hpFields)
    hpKey = hpFields{i};
    if isempty(HP.(hpKey))
        continue;
    end
    
    try
        T = unstack(HP.(hpKey), 'Count', 'Value', 'AggregationFunction', @sum);
        writetable(T, fullfile(outTabDir, sprintf('TableS1_%s_Stability.csv', hpKey)));
        table2latex(T, fullfile(outTabDir, sprintf('TableS1_%s_Stability.tex', hpKey)), ...
            'Caption', sprintf('Stability of selected %s across outer folds.', hpKey), ...
            'Label', sprintf('tab:%s_stability', lower(hpKey)));
    catch ME
        warning('Could not create hyperparameter table for %s: %s', hpKey, ME.message);
    end
end

%% === Load Phase 3 ===
if ~exist(phase3File, 'file')
    warning('Phase 3 file not found: %s. Skipping Phase 3 analysis.', phase3File);
    fprintf('\nDone. Tables in "%s", figures in "%s".\n', outTabDir, outFigDir);
    return;
end

S3 = load(phase3File);
if ~isfield(S3, 'results')
    warning('Phase 3 file missing results field. Skipping Phase 3 analysis.');
    fprintf('\nDone. Tables in "%s", figures in "%s".\n', outTabDir, outFigDir);
    return;
end

R = S3.results(:);
P3 = numel(R);
pipeNames3 = strings(P3,1);
MET_CV = nan(P3, numel(metricNames));
MET_PROBE = nan(P3, numel(metricNames));
hasProbe = false;

for i = 1:P3
    if isfield(R(i), 'name')
        pipeNames3(i) = string(R(i).name);
    else
        pipeNames3(i) = sprintf("Pipeline_%d", i);
    end
    
    % metrics (CV)
    if isfield(R(i), 'metrics')
        try
            MET_CV(i,:) = struct2array(orderfields(R(i).metrics, orderfields_from(metricNames)));
        catch
            warning('Could not extract CV metrics for pipeline %d', i);
        end
    end
    
    % probeMetrics (if present)
    if isfield(R(i), 'probeMetrics') && isstruct(R(i).probeMetrics)
        try
            MET_PROBE(i,:) = struct2array(orderfields(R(i).probeMetrics, orderfields_from(metricNames)));
            hasProbe = true;
        catch
            warning('Could not extract probe metrics for pipeline %d', i);
        end
    end
end

% === Table 2: Probe set comparison ===
if hasProbe && any(~isnan(MET_PROBE(:)))
    T2 = array2table(MET_PROBE, 'VariableNames', cellstr(metricNames));
    T2 = addvars(T2, pipeNames3, 'Before', 1, 'NewVariableNames','Pipeline');
    writetable(T2, fullfile(outTabDir,'Table2_Probe_Comparison.csv'));

    % Format table for LaTeX output
    T2_tex = T2;
    for m = 1:numel(metricNames)
        metric_col = MET_PROBE(:,m);
        valid_idx = ~isnan(metric_col);
        if any(valid_idx)
            [~,ix] = max(metric_col(valid_idx));
            % Find the actual index in the original array
            valid_indices = find(valid_idx);
            best_idx = valid_indices(ix);

            % Convert entire column to formatted strings in a cell array
            formatted_col = cell(size(metric_col));
            for j = 1:length(metric_col)
                if isnan(metric_col(j))
                    formatted_col{j} = 'NaN';
                else
                    formatted_col{j} = sprintf('%.3f', metric_col(j));
                end
            end
            
            % Bold the best one
            if ~isnan(metric_col(best_idx))
                formatted_col{best_idx} = sprintf('\\textbf{%.3f}', metric_col(best_idx));
            end

            % Place the formatted cell column back into the table
            T2_tex.(metricNames(m)) = formatted_col;
        end
    end
    
    table2latex(T2_tex, fullfile(outTabDir,'Table2_Probe_Comparison.tex'), ...
        'Caption', 'Performance on held-out probe set.', ...
        'Label', 'tab:probe_comparison');
end

%% === Confusion matrices & ROC curves for each model (Phase 3) ===
for i = 1:P3
    mdlName = pipeNames3(i);

    if isfield(R(i),'probeTable') && istable(R(i).probeTable) && ...
            all(ismember({'TrueLabel','PredLabel','MeanProbWHO3'}, R(i).probeTable.Properties.VariableNames))
        % Confusion matrix from probe-level predictions
        yTrue = R(i).probeTable.TrueLabel;
        yPred = R(i).probeTable.PredLabel;
        C = confusionmat(yTrue, yPred, 'Order',[1 3]);
        plot_confusion_matrix(C, fullfile(outFigDir, sprintf('Confusion_%s.pdf', mdlName)));

        % ROC curve using probe-level probabilities
        [Xroc,Yroc,~,AUC] = perfcurve(yTrue, R(i).probeTable.MeanProbWHO3, 3);
        fig = figure('Units','centimeters','Position',[2 2 8 6]);
        plot(Xroc,Yroc,'LineWidth',1.5); grid on; axis square
        xlabel('False positive rate'); ylabel('True positive rate');
        title(sprintf('ROC - %s (AUC %.3f)', mdlName, AUC));
        exportgraphics(fig, fullfile(outFigDir, sprintf('ROC_%s.pdf', mdlName)), 'ContentType','vector');
        close(fig);
    end
end

%% === Optional: Paired significance test across pipelines (outer folds) ===
% Example: Wilcoxon signed-rank test comparing AUC distributions across folds
if ~isempty(aucIdx) && any(strcmp(pipeNames, "BaselineLDA"))
    fprintf('\nPaired Wilcoxon on outer-fold AUC (BaselineLDA vs others):\n');
    base_idx = find(strcmp(pipeNames, "BaselineLDA"), 1);
    base = RAW{base_idx}(:,aucIdx);
    
    for p = 1:P
        if strcmp(pipeNames(p), "BaselineLDA"), continue; end
        try
            pval = signrank(base, RAW{p}(:,aucIdx)); % non-parametric paired
            fprintf('  vs %-10s : p = %.4f\n', pipeNames(p), pval);
        catch ME
            warning('Could not perform signrank test for %s: %s', pipeNames(p), ME.message);
        end
    end
end

fprintf('\nDone. Tables in "%s", figures in "%s".\n', outTabDir, outFigDir);

end

%% === Helpers ===
function s = orderfields_from(metricNames)
    % Returns a struct with matching metric order to use with orderfields
    s = struct();
    for i = 1:numel(metricNames)
        s.(char(metricNames(i))) = []; % Convert string to char for fieldname
    end
end

function table2latex(T, outFile, varargin)
% Minimal table->LaTeX writer (no dependencies).
% Usage: table2latex(T, 'out.tex', 'Caption','...', 'Label','...')
p = inputParser; 
addParameter(p, 'Caption', ''); 
addParameter(p, 'Label', ''); 
parse(p, varargin{:});
cap = p.Results.Caption; 
lab = p.Results.Label;

fid = fopen(outFile,'w');
if fid == -1
    error('Cannot open file for writing: %s', outFile);
end

fprintf(fid, '%% Auto-generated by make_paper_figs_and_tables.m\n');
fprintf(fid, '\\begin{table}[t]\\centering\n\\small\n');
fprintf(fid, '\\begin{tabular}{l%s}\\toprule\n', repmat('c',1, width(T)-1));

% header
fprintf(fid, '%s', T.Properties.VariableNames{1});
for j = 2:width(T)
    fprintf(fid, ' & %s', strrep(T.Properties.VariableNames{j},'_','\\_'));
end
fprintf(fid, ' \\\\ \\midrule\n');

% rows
for i = 1:height(T)
    for j = 1:width(T)
        cellVal = T{i,j};
        % If the cell already contains a cell (from pre-formatting), extract its content
        if iscell(cellVal) && ~isempty(cellVal)
            cellVal = cellVal{1}; 
        end

        if ischar(cellVal) || isstring(cellVal)
            txt = string(cellVal);
        elseif isnumeric(cellVal) && ~isnan(cellVal)
            txt = sprintf('%.3f', cellVal);
        else
            txt = 'NaN';
        end
        
        if j == 1
            fprintf(fid, '%s', txt);
        else
            fprintf(fid, ' & %s', txt);
        end
    end
    fprintf(fid, ' \\ \n');
end

fprintf(fid, '\\bottomrule\n\\end{tabular}\n');
if ~isempty(cap), fprintf(fid, '\\caption{%s}\n', cap); end
if ~isempty(lab), fprintf(fid, '\\label{%s}\n', lab); end
fprintf(fid, '\\end{table}\n');
fclose(fid);
end

function plot_confusion_matrix(C, outFile)
    if ~ismatrix(C) || ~all(size(C)==[2 2])
        warning('Confusion matrix must be 2x2. Skipping plot.'); 
        return;
    end
    
    if sum(C(:)) == 0
        warning('Confusion matrix is empty. Skipping plot.');
        return;
    end
    
    P = C ./ sum(C,2); % row-normalized
    P(isnan(P)) = 0; % Handle division by zero
    
    fig = figure('Units','centimeters','Position',[2 2 8 7]);
    imagesc(P, [0 1]); axis image; colormap(parula); colorbar
    xticks(1:2); yticks(1:2);
    xticklabels({'Pred 0','Pred 1'}); yticklabels({'True 0','True 1'});
    
    for i = 1:2
        for j = 1:2
            text(j,i, sprintf('%d\n(%.2f)', C(i,j), P(i,j)), ...
                'HorizontalAlignment','center','Color','k','FontWeight','bold');
        end
    end
    
    title('Confusion Matrix (counts and row-normalized)')
    exportgraphics(fig, outFile, 'ContentType','vector');
    close(fig);
end
