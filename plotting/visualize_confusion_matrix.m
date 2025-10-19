function visualize_confusion_matrix(cfg, opts)
%VISUALIZE_CONFUSION_MATRIX Display confusion matrices for Phases 2 and 3 predictions.
%
%   VISUALIZE_CONFUSION_MATRIX(cfg, opts) searches for the most recent
%   Phase 3 comparison results file, loads probe-level predictions for the
%   best pipeline from Phase 2 (cross-validation) and Phase 3 (test),
%   computes confusion matrices, and displays them using CONFUSIONCHART
%   with human-readable class labels.
%
%   cfg  - configuration struct from CONFIGURE_CFG (optional)
%   opts - plotting options from PLOT_SETTINGS (optional)
%
%   Example:
%       visualize_confusion_matrix();
%       cfg = configure_cfg('projectRoot', '/path/to/project');
%       visualize_confusion_matrix(cfg);
%
%   Date: 2025-06-11

    if nargin < 1 || isempty(cfg)
        cfg = configure_cfg();
    end
    if nargin < 2 || isempty(opts)
        opts = plot_settings(); %#ok<NASGU> % opts currently unused but kept for consistency
    end

    P = setup_project_paths(cfg.projectRoot, '', cfg);

    % Locate latest Phase 3 comparison results file
    p3_dir = fullfile(P.resultsPath, 'Phase3');
    compFiles = dir(fullfile(p3_dir, '*_Phase3_ComparisonResults.mat'));
    if isempty(compFiles)
        error('No Phase 3 comparison results file found in %s.', p3_dir);
    end
    [~, idx] = sort([compFiles.datenum], 'descend');
    resultsFile = fullfile(compFiles(idx(1)).folder, compFiles(idx(1)).name);
    S = load(resultsFile, 'bestModelInfo');
    if ~isfield(S, 'bestModelInfo')
        error('bestModelInfo not found in %s.', resultsFile);
    end
    bmi = S.bestModelInfo;

    classOrder = [1 3];
    classNames = {'WHO-1','WHO-3'};

    %% Phase 3 predictions (test set)
    if ~isfield(bmi, 'probeTable')
        error('probeTable with Phase 3 predictions not found in bestModelInfo.');
    end
    tblP3 = bmi.probeTable;
    yTrueP3 = tblP3.TrueLabel;
    yPredP3 = tblP3.PredLabel;
    cmP3 = confusionmat(yTrueP3, yPredP3, 'Order', classOrder); %#ok<NASGU>
    figure('Name', 'Confusion Matrix - Phase 3');
    confusionchart(categorical(yTrueP3, classOrder, classNames), ...
                   categorical(yPredP3, classOrder, classNames));
    title(sprintf('Confusion Matrix (Phase 3) - %s', bmi.name));

    %% Phase 2 predictions (cross-validation) - optional
    tblP2 = [];
    if isfield(bmi, 'probeTableCV')
        tblP2 = bmi.probeTableCV;
    elseif isfield(bmi, 'probeTablePhase2')
        tblP2 = bmi.probeTablePhase2;
    elseif isfield(bmi, 'phase2ProbeTable')
        tblP2 = bmi.phase2ProbeTable;
    end

    if ~isempty(tblP2)
        yTrueP2 = tblP2.TrueLabel;
        yPredP2 = tblP2.PredLabel;
        cmP2 = confusionmat(yTrueP2, yPredP2, 'Order', classOrder); %#ok<NASGU>
        figure('Name', 'Confusion Matrix - Phase 2');
        confusionchart(categorical(yTrueP2, classOrder, classNames), ...
                       categorical(yPredP2, classOrder, classNames));
        title(sprintf('Confusion Matrix (Phase 2) - %s', bmi.name));
    else
        warning('Phase 2 predictions not found in %s. Only Phase 3 confusion matrix displayed.', resultsFile);
    end
end

