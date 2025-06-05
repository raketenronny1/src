% main.m
%
% Simple wrapper script to run the key project phases using default
% configuration settings.
%
% Users can modify the `cfg` structure below to customize paths or
% parameters and then run this file for a one-click execution.

cfg = struct();

run_phase2_model_selection_comparative(cfg);
run_phase3_final_evaluation(cfg);
run_phase4_feature_interpretation(cfg);
