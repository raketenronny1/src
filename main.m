% main.m
%
% Simple wrapper script to run the key project phases using default
% configuration settings.
%
% Users can modify the configuration via `configure_cfg` or by editing the
% returned struct directly.

cfg = configure_cfg();

run_phase2_model_selection_comparative(cfg);
run_phase3_final_evaluation(cfg);
run_phase4_feature_interpretation(cfg);
