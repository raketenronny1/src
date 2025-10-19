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

    cfg = resolve_cfg_input(nargin, cfgInput);

    run_phase2_model_selection(cfg);
    run_phase3_final_evaluation(cfg);
    run_phase4_feature_interpretation(cfg);
end

function cfg = resolve_cfg_input(narginValue, cfgInput)
    if narginValue == 0 || isempty(cfgInput)
        cfg = configure_cfg();
    elseif isstruct(cfgInput)
        cfg = configure_cfg(cfgInput);
    elseif ischar(cfgInput) || (isstring(cfgInput) && isscalar(cfgInput))
        cfg = configure_cfg('configFile', char(cfgInput));
    else
        error('main:InvalidConfigInput', ...
            'Config input must be empty, a struct or a file path.');
    end
end
