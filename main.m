function main(cfgInput)
%MAIN Execute the core project phases with the supplied configuration.
%
%   MAIN() loads config/default.yaml and runs phases 2–4 sequentially.
%   MAIN(configPath) loads overrides from the provided YAML file before
%   executing the phases.
%   MAIN(cfgStruct) accepts an existing configuration struct.
%
%   Example:
%       main('config/custom.yaml');

helperPath = fullfile(fileparts(mfilename('fullpath')), 'helper_functions');
if exist('configure_cfg','file') ~= 2 && isfolder(helperPath)
    addpath(helperPath);
end

cfg = configure_cfg();

    run_phase2_model_selection(cfg);
    run_phase3_final_evaluation(cfg);
    run_phase4_feature_interpretation(cfg);
end

cfg = validate_configuration(cfg);

run_phase2_model_selection(cfg);
run_phase3_final_evaluation(cfg);
run_phase4_feature_interpretation(cfg);
