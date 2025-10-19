function visualize_model_comparison_spiderplots(cfg, opts, results)
%VISUALIZE_MODEL_COMPARISON_SPIDERPLOTS Create spider plots for AUC and F2\_WHO3 across models.
%
%   visualize_model_comparison_spiderplots(cfg, opts) loads the latest
%   Phase 3 comparison results and produces two spider plots comparing
%   model performance for the metrics AUC and F2\_WHO3. Figures are saved
%   under figures/Phase3/ModelComparison.
%
%   cfg     - configuration struct from CONFIGURE_CFG (optional)
%   opts    - plotting options from PLOT_SETTINGS (optional)
%   results - optional struct array with fields .name and .metrics.
%             When provided, this data is used instead of loading from file.

    if nargin < 1 || isempty(cfg)
        cfg = configure_cfg();
    end
    if nargin < 2 || isempty(opts)
        opts = plot_settings();
    end

    P = setup_project_paths(cfg.projectRoot, '', cfg);

    if nargin < 3 || isempty(results)
        resDir = fullfile(P.resultsPath, 'Phase3');
        files = dir(fullfile(resDir, '*_Phase3_ComparisonResults.mat'));
        if isempty(files)
            error('No Phase 3 comparison results found in %s.', resDir);
        end
        [~, idx] = sort([files.datenum], 'descend');
        S = load(fullfile(files(idx(1)).folder, files(idx(1)).name), 'results');
        results = S.results;
    end

    modelNames = {results.name};
    numModels = numel(modelNames);

    aucVals = arrayfun(@(r) r.metrics.AUC, results);
    f2Vals  = arrayfun(@(r) r.metrics.F2_WHO3, results);

    axesLimits = repmat([0.80; 1], 1, numModels);

    outDir = fullfile(P.figuresPath, 'Phase3', 'ModelComparison');
    if ~isfolder(outDir), mkdir(outDir); end

    %% AUC spider plot
    figAUC = figure('Name', 'Model Comparison - AUC');
 spider_plot_R2019b(aucVals, 'AxesLabels', modelNames, 'AxesLimits', axesLimits, ...
    'AxesInterval', 5, 'AxesPrecision', 2, ...
    'FillOption', 'on', 'FillTransparency', 0.1, ...
    'Color', opts.colorTest, 'LineWidth', 2, ...
    'Marker', 'o', 'MarkerSize', 6);
    title('Model Comparison: AUC');
    outBase = fullfile(outDir, sprintf('%s_P3_ModelComparison_AUC', opts.datePrefix));
    savefig(figAUC, [outBase '.fig']);
    if exist('exportgraphics', 'file') == 2
        exportgraphics(figAUC, [outBase '.tiff'], 'Resolution', 300);
    else
        print(figAUC, [outBase '.tiff'], '-dtiff', '-r300');
    end

    %% F2\_WHO3 spider plot
    figF2 = figure('Name', 'Model Comparison - F2\\_WHO3');
   spider_plot_R2019b(f2Vals, 'AxesLabels', modelNames, 'AxesLimits', axesLimits, ...
    'AxesInterval', 5, 'AxesPrecision', 2, ...
    'FillOption', 'on', 'FillTransparency', 0.1, ...
    'Color', opts.colorTest, 'LineWidth', 2, ...
    'Marker', 's', 'MarkerSize', 6);
    outBase = fullfile(outDir, sprintf('%s_P3_ModelComparison_F2WHO3', opts.datePrefix));
    savefig(figF2, [outBase '.fig']);
    if exist('exportgraphics', 'file') == 2
        exportgraphics(figF2, [outBase '.tiff'], 'Resolution', 300);
    else
        print(figF2, [outBase '.tiff'], '-dtiff', '-r300');
    end

    fprintf('Saved model comparison spider plots to %s\n', outDir);
end
