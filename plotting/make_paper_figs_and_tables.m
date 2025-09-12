function make_paper_figs_and_tables
% Create paper-ready tables and figures from Phase 2/3 results.
% Outputs:
%   tables/Table1_CV_Comparison.csv / .tex
%   tables/Table2_Probe_Comparison.csv / .tex
%   tables/TableS1_HyperparamStability.csv / .tex
%   figures/Fig1_AUC_withErrorBars.pdf (vector)
%   figures/FigS1_AllMetrics_withErrorBars.pdf (vector)

clc; close all; rng default
set(groot,'defaultAxesFontName','Helvetica','defaultAxesFontSize',9);
set(groot,'defaultTextInterpreter','none'); % keep labels literal

%% === Paths (edit if needed) ===
phase2File = '20250912_Phase2_AllPipelineResults.mat';
phase3File = '20250912_Phase3_ComparisonResults.mat';
outTabDir  = fullfile(pwd,'tables');   if ~exist(outTabDir,'dir'), mkdir(outTabDir); end
outFigDir  = fullfile(pwd,'figures');  if ~exist(outFigDir,'dir'), mkdir(outFigDir); end

%% === Load Phase 2 ===
S2 = load(phase2File);
metricNames = string(S2.metricNames(:)');
Kouter = S2.numOuterFolds; Kinner = S2.numInnerFolds;

% Collect mean and per-fold metrics for each pipeline
P = numel(S2.resultsPerPipeline);
pipeNames = strings(P,1);
MEAN = nan(P, numel(metricNames));
SD   = nan(P, numel(metricNames));
RAW  = cell(P,1);
for p = 1:P
    r = S2.resultsPerPipeline(p);
    pipeNames(p) = string(r.pipelineConfig.name);
    RAW{p}      = r.outerFoldMetrics_raw;               % (Kouter x M)
    MEAN(p,:)   = r.outerFoldMetrics_mean(:)';
    SD(p,:)     = std(RAW{p}, 0, 1, 'omitnan');         % sample SD across outer folds
end

% === Table 1: Nested CV comparison (mean ± SD) ===
fmt = @(m,s) arrayfun(@(mi,si) sprintf('%.3f ± %.3f', mi, si), m, s, 'uni', 0);
T1 = table(pipeNames, 'VariableNames', {'Pipeline'});
for m = 1:numel(metricNames)
    T1.(metricNames(m)) = fmt(MEAN(:,m), SD(:,m))';
end
% Bold best per metric (higher is better) using LaTeX mark-up
T1_tex = T1;
for m = 1:numel(metricNames)
    [~,ix] = max(MEAN(:,m));
    T1_tex{ix, m+1} = {['\textbf{', T1_tex{ix, m+1}{1}, '}']};
end
writetable(T1, fullfile(outTabDir,'Table1_CV_Comparison.csv'));
table2latex(T1_tex, fullfile(outTabDir,'Table1_CV_Comparison.tex'), ...
    'Caption', sprintf('Nested %dx%d CV performance (mean ± SD).', Kouter, Kinner), ...
    'Label', 'tab:cv_comparison');

% === Fig 1: AUC with error bars across pipelines ===
[~, aucIdx] = ismember("AUC", metricNames);
if aucIdx>0
    fig = figure('Units','centimeters','Position',[2 2 9 6]);
    hold on
    x = 1:P;
    hB = bar(x, MEAN(:,aucIdx)); %#ok<NASGU>
    er = errorbar(x, MEAN(:,aucIdx), SD(:,aucIdx), 'LineStyle','none');
    xticks(x); xticklabels(pipeNames); xtickangle(20)
    ylabel('AUC'); ylim([0 1]); grid on
    title('Cross-validated AUC by pipeline')
    exportgraphics(fig, fullfile(outFigDir,'Fig1_AUC_withErrorBars.pdf'), 'ContentType','vector');
end

% === Optional: All metrics with error bars (supplement) ===
fig = figure('Units','centimeters','Position',[2 2 14 9]);
tiledlayout('flow')
for m = 1:numel(metricNames)
    nexttile
    bar(MEAN(:,m)); hold on
    errorbar(1:P, MEAN(:,m), SD(:,m), 'LineStyle','none');
    title(metricNames(m)); ylim([0 1]); grid on
    xticks(1:P); xticklabels(pipeNames); xtickangle(30)
end
exportgraphics(fig, fullfile(outFigDir,'FigS1_AllMetrics_withErrorBars.pdf'), 'ContentType','vector');

% === Table S1: Hyperparameter stability across folds ===
HP = struct();  % struct of arrays
for p = 1:P
    hpCells = S2.resultsPerPipeline(p).outerFoldBestHyperparams;
    names = fieldnames(hpCells(1));
    for k = 1:numel(names)
        v = arrayfun(@(h) h.(names{k}), hpCells);
        key = names{k};
        if ~isfield(HP, key), HP.(key) = table(); end
        % Count frequencies per pipeline
        cats = unique(v);
        cnts = cellfun(@(c) nnz(isequaln(v, c)), num2cell(cats)); %#ok<DISEQN>
        % robust count (fallback)
        cnts = arrayfun(@(c) sum(v==c), cats);
        T = table(string(pipeNames(p)), categorical(cats(:)), cnts(:), ...
            'VariableNames', {'Pipeline', 'Value', 'Count'});
        HP.(key) = [HP.(key); T];
    end
end
% Write all HP tables
hpFields = fieldnames(HP);
for i = 1:numel(hpFields)
    hpKey = hpFields{i};
    T = unstack(HP.(hpKey), 'Count', 'Value', 'AggregationFunction', @sum);
    writetable(T, fullfile(outTabDir, sprintf('TableS1_%s_Stability.csv', hpKey)));
    table2latex(T, fullfile(outTabDir, sprintf('TableS1_%s_Stability.tex', hpKey)), ...
        'Caption', sprintf('Stability of selected %s across outer folds.', hpKey), ...
        'Label', sprintf('tab:%s_stability', lower(hpKey)));
end

%% === Load Phase 3 ===
S3 = load(phase3File);
R = S3.results(:);
P3 = numel(R);
pipeNames3 = strings(P3,1);
MET_CV = nan(P3, numel(metricNames));
MET_PROBE = nan(P3, numel(metricNames));
hasProbe = true;

for i = 1:P3
    pipeNames3(i) = string(R(i).name);
    % metrics (CV)
    MET_CV(i,:) = struct2array(orderfields(R(i).metrics, orderfields_from(metricNames)));
    % probeMetrics (if present)
    if isfield(R, 'probeMetrics') && isstruct(R(i).probeMetrics)
        MET_PROBE(i,:) = struct2array(orderfields(R(i).probeMetrics, orderfields_from(metricNames)));
    else
        hasProbe = false;
    end
end

% === Table 2: Probe set comparison ===
if hasProbe
    T2 = array2table(MET_PROBE, 'VariableNames', cellstr(metricNames));
    T2 = addvars(T2, pipeNames3, 'Before', 1, 'NewVariableNames','Pipeline');
    % bold best per metric
    T2_tex = T2;
    for m = 1:numel(metricNames)
        [~,ix] = max(MET_PROBE(:,m));
        T2_tex{ix, m+1} = {sprintf('\\textbf{%.3f}', MET_PROBE(ix,m))};
        for r = setdiff(1:P3, ix)'
            T2_tex{r, m+1} = {sprintf('%.3f', MET_PROBE(r,m))};
        end
    end
    writetable(T2, fullfile(outTabDir,'Table2_Probe_Comparison.csv'));
    table2latex(T2_tex, fullfile(outTabDir,'Table2_Probe_Comparison.tex'), ...
        'Caption', 'Performance on held-out probe set.', ...
        'Label', 'tab:probe_comparison');
end

%% === Optional: Confusion matrix & ROC for best model (Phase 3) ===
if isfield(S3, 'bestModelInfo')
    B = S3.bestModelInfo;
    % Confusion matrix from predicted + (if available) true labels
    yhat = []; ytrue = [];
    if isfield(B,'predicted'), yhat = double(B.predicted(:)); end
    % Try to fetch ytrue from probeTable (if it stores labels)
    if isfield(B,'probeTable') && isnumeric(B.probeTable)
        % If probeTable is 2x2 confusion matrix:
        C = B.probeTable;
        plot_confusion_matrix(C, fullfile(outFigDir,'Fig2_Best_ConfusionMatrix.pdf'));
    elseif ~isempty(yhat)
        % If you can provide ytrue, do it here:
        % ytrue = ...; % [N x 1] logical or 0/1
        % If available:
        % C = confusionmat(ytrue, yhat);
        % plot_confusion_matrix(C, fullfile(outFigDir,'Fig2_Best_ConfusionMatrix.pdf'));
    end

    % ROC from saved file or recompute if scores+ytrue available
    if isfield(B,'rocFile') && isfile(B.rocFile)
        % Just copy existing ROC image alongside figures
        copyfile(B.rocFile, fullfile(outFigDir,'Fig3_Best_ROC.png'));
    elseif isfield(B,'scores') && ~isempty(ytrue)
        [X,Y,~,AUC] = perfcurve(ytrue, B.scores(:), 1);
        fig = figure('Units','centimeters','Position',[2 2 8 6]);
        plot(X,Y,'LineWidth',1.5); grid on; axis square
        xlabel('1 - Specificity'); ylabel('Sensitivity'); title(sprintf('ROC (AUC=%.3f)',AUC));
        exportgraphics(fig, fullfile(outFigDir,'Fig3_Best_ROC.pdf'), 'ContentType','vector');
    end
end

%% === Optional: Paired significance test across pipelines (outer folds) ===
% Example: Wilcoxon signed-rank test comparing AUC distributions across folds
if aucIdx>0
    fprintf('\nPaired Wilcoxon on outer-fold AUC (BaselineLDA vs others):\n');
    base = RAW{pipeNames=="BaselineLDA"}(:,aucIdx);
    for p = 1:P
        if pipeNames(p)=="BaselineLDA", continue; end
        pval = signrank(base, RAW{p}(:,aucIdx)); % non-parametric paired
        fprintf('  vs %-10s : p = %.4f\n', pipeNames(p), pval);
    end
end

fprintf('\nDone. Tables in "%s", figures in "%s".\n', outTabDir, outFigDir);

end

%% === Helpers ===
function s = orderfields_from(metricNames)
    % Returns a struct with matching metric order to use with orderfields
    for i = 1:numel(metricNames)
        s.(metricNames{i}) = [];
    end
end

function table2latex(T, outFile, varargin)
% Minimal table->LaTeX writer (no dependencies).
% Usage: table2latex(T, 'out.tex', 'Caption','...', 'Label','...')
p = inputParser; p.addParameter('Caption',''); p.addParameter('Label',''); p.parse(varargin{:});
cap = p.Results.Caption; lab = p.Results.Label;

fid = fopen(outFile,'w');
fprintf(fid, '%% Auto-generated by make_paper_figs_and_tables.m\n');
fprintf(fid, '\\begin{table}[t]\\centering\n\\small\n');
fprintf(fid, '\\begin{tabular}{l%s}\\toprule\n', repmat('c',1, width(T)-1));
% header
fprintf(fid, '%s', T.Properties.VariableNames{1});
for j = 2:width(T)
    fprintf(fid, ' & %s', strrep(T.Properties.VariableNames{j},'_','\\_'));
end
fprintf(fid, ' \\ \\midrule\n');
% rows
for i = 1:height(T)
    fprintf(fid, '%s', string(table2cell(T(i,1))));
    for j = 2:width(T)
        cellstrVal = table2cell(T(i,j));
        if ischar(cellstrVal{1}) || isstring(cellstrVal{1})
            txt = string(cellstrVal{1});
        else
            txt = sprintf('%.3f', cellstrVal{1});
        end
        fprintf(fid, ' & %s', txt);
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
        warning('Confusion matrix must be 2x2. Skipping plot.'); return;
    end
    P = C ./ sum(C,2); % row-normalized
    fig = figure('Units','centimeters','Position',[2 2 8 7]);
    imagesc(P, [0 1]); axis image; colormap(parula); colorbar
    xticks(1:2); yticks(1:2);
    xticklabels({'Pred 0','Pred 1'}); yticklabels({'True 0','True 1'});
    for i = 1:2, for j = 1:2
        text(j,i, sprintf('%d\n(%.2f)', C(i,j), P(i,j)), ...
            'HorizontalAlignment','center','Color','k','FontWeight','bold');
    end, end
    title('Confusion Matrix (counts and row-normalized)')
    exportgraphics(fig, outFile, 'ContentType','vector');
end
