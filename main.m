% main.m
%
% Simple wrapper script to run the key project phases using default
% configuration settings.
%
% Users can modify the configuration via `configure_cfg` or by editing the
% returned struct directly.

cfg = configure_cfg();

disp('Select outlier removal strategy for evaluation:');
disp(' 1 - AND (consensus, recommended)');
disp(' 2 - OR  (lenient)');
usr = input('Enter choice [1]: ','s');
if isempty(usr) || str2double(usr)==1
    cfg.outlierStrategy = 'AND';
else
    cfg.outlierStrategy = 'OR';
end

% Ensure Phase 2 compares only the selected strategy
cfg.outlierStrategiesToCompare = {cfg.outlierStrategy};

run_phase2_model_selection_comparative(cfg);
run_phase3_final_evaluation(cfg);
run_phase4_feature_interpretation(cfg);
