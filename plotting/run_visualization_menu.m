function run_visualization_menu()
%RUN_VISUALIZATION_MENU Interactive helper to call visualization functions.
%
%   Presents a simple menu to generate various plots for the project.

    cfg  = configure_cfg();
    opts = plot_settings();
    P    = setup_project_paths(cfg.projectRoot);

    choice = -1;
    while choice ~= 0
        fprintf('\nSelect visualization to generate:\n');
        fprintf(' 1 - Phase 1 figures\n');
        fprintf(' 2 - Exploratory outlier visualizations\n');
        fprintf(' 3 - Binning effect plot\n');
        fprintf(' 4 - Project summary (Phases 2-4)\n');
        fprintf(' 5 - Phase 2 fold metrics\n');
        fprintf(' 6 - Confusion matrix (Phases 2 & 3)\n');
        fprintf(' 0 - Exit\n');
        usr = input('Enter choice: ','s');
        if isempty(usr), choice = 0; else choice = str2double(usr); end
        switch choice
            case 1
                visualize_phase1(P, opts);
            case 2
                warning('Outlier visualization requires many inputs and is typically called from preprocessing scripts.');
            case 3
                visualize_binning_effects(P, opts);
            case 4
                visualize_project_summary(cfg, opts);
            case 5
                visualize_fold_metrics(P, opts);
            case 6
                visualize_confusion_matrix(cfg, opts);
            otherwise
                if choice ~= 0
                    disp('Invalid selection.');
                end
        end
    end
end
