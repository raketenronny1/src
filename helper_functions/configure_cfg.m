function cfg = configure_cfg(varargin)
%CONFIGURE_CFG Create or update configuration struct for project scripts.
%
%   cfg = CONFIGURE_CFG() returns a struct with default fields.
%   cfg = CONFIGURE_CFG(existingCfg) fills missing fields in existingCfg.
%   cfg = CONFIGURE_CFG(...,'Name',Value) sets or overrides fields.
%
%   Recognised fields:
%     projectRoot                - repository root path
%     outlierStrategy            - placeholder for compatibility
%     outlierStrategiesToCompare - placeholder array
%
%   Example:
%       cfg = configure_cfg('projectRoot','/path/to/project', ...
%                           'outlierStrategy','OR');
%
% Date: 2025-06-06

    if nargin > 0 && isstruct(varargin{1})
        cfg = varargin{1};
        args = varargin(2:end);
    else
        cfg = struct();
        args = varargin;
    end

    if mod(numel(args),2) ~= 0
        error('Name-value arguments must occur in pairs.');
    end
    for i = 1:2:numel(args)
        name = args{i};
        value = args{i+1};
        cfg.(name) = value;
    end

    if ~isfield(cfg,'projectRoot') || isempty(cfg.projectRoot)
        if exist('get_project_root','file')
            cfg.projectRoot = get_project_root();
        else
            cfg.projectRoot = pwd;
        end
    end
    if ~isfield(cfg,'outlierStrategy')
        cfg.outlierStrategy = 'default';
    end
    if ~isfield(cfg,'outlierStrategiesToCompare')
        cfg.outlierStrategiesToCompare = {'default'};
    end
end

