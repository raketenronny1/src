% main.m
%
% Simple wrapper script to run the key project phases using default
% configuration settings.
%
% Users can modify the configuration via `configure_cfg` or by editing the
% returned struct directly.

cfg = configure_cfg();

% Use the recommended AND strategy by default.  Edit `cfg.outlierStrategy`
% manually if a different behaviour is desired.
cfg.outlierStrategy = 'AND';

% Ensure Phase 2 compares only the selected strategy
cfg.outlierStrategiesToCompare = {cfg.outlierStrategy};

run_phase2_model_selection_comparative(cfg);
run_phase3_final_evaluation(cfg);
run_phase4_feature_interpretation(cfg);
