% main.m
%
% Simple wrapper script to run the key project phases using default
% configuration settings.
%
% Users can modify the configuration via `configure_cfg` or by editing the
% returned struct directly.

helperPath = fullfile(fileparts(mfilename('fullpath')), 'helper_functions');
if exist('configure_cfg','file') ~= 2 && isfolder(helperPath)
    addpath(helperPath);
end

cfg = configure_cfg();

% Choose whether to run Phase 2 on the outlier-filtered training set or on
% the unaltered data.
cfg.useOutlierRemoval = false;  % set to false to analyse the full dataset

cfg = validate_configuration(cfg);

run_phase2_model_selection(cfg);
run_phase3_final_evaluation(cfg);
run_phase4_feature_interpretation(cfg);
