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
%
%   The helper path is added dynamically so the script can be executed from
%   any working directory inside the repository.

helperPath = fullfile(fileparts(mfilename('fullpath')), 'helper_functions');
if exist('configure_cfg','file') ~= 2 && isfolder(helperPath)
    addpath(helperPath);
end

if nargin < 1 || isempty(cfgInput)
    cfgArgs = {};
elseif isstruct(cfgInput)
    cfgArgs = {cfgInput};
elseif isstring(cfgInput) || ischar(cfgInput)
    cfgArgs = {'configFile', char(cfgInput)};
else
    error('main:InvalidInput', ['Unsupported configuration input. Pass a struct, a file path, or leave empty to use defaults. ', ...
        'Troubleshooting tip: check the call site of main.m.']);
end

cfg = configure_cfg(cfgArgs{:});
cfg = validate_configuration(cfg);

run_phase2_model_selection(cfg);
run_phase3_final_evaluation(cfg);
run_phase4_feature_interpretation(cfg);
end
