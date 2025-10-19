function cfg = configure_cfg(varargin)
%CONFIGURE_CFG Create or update configuration struct for project scripts.
%
%   cfg = CONFIGURE_CFG() returns a struct with default fields.
%   cfg = CONFIGURE_CFG(existingCfg) fills missing fields in existingCfg.
%   cfg = CONFIGURE_CFG(...,'Name',Value) sets or overrides fields.
%
%   Recognised fields:
%     projectRoot        - repository root path
%     useOutlierRemoval  - true to load pre-filtered training data
%     chunkSizes         - struct controlling chunk-aware helpers:
%         .flattenSpectra  - number of probes to process per batch when
%                            flattening spectra (default [] -> all at once).
%         .binSpectraRows  - number of spectra rows per binning pass.
%         .fisherPerClass  - rows per class batch when computing Fisher ratios.
%
%   Smaller chunk sizes reduce peak memory consumption at the cost of extra
%   loop overhead. Leaving the value empty ([]) processes the entire dataset
%   in one pass, matching legacy behaviour.
%
%   Example:
%       cfg = configure_cfg('projectRoot','/path/to/project', ...
%                           'useOutlierRemoval',true);
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
    if ~isfield(cfg,'useOutlierRemoval')
        cfg.useOutlierRemoval = true;
    end
    if ~isfield(cfg,'parallelOutlierComparison')
        cfg.parallelOutlierComparison = true;
    end
    if ~isfield(cfg,'outlierAlpha')
        cfg.outlierAlpha = 0.01;
    end
    if ~isfield(cfg,'outlierVarianceToModel')
        cfg.outlierVarianceToModel = 0.95;
    end
    if ~isfield(cfg,'chunkSizes') || ~isstruct(cfg.chunkSizes)
        cfg.chunkSizes = struct();
    end
    if ~isfield(cfg.chunkSizes,'flattenSpectra')
        cfg.chunkSizes.flattenSpectra = [];
    end
    if ~isfield(cfg.chunkSizes,'binSpectraRows')
        cfg.chunkSizes.binSpectraRows = [];
    end
    if ~isfield(cfg.chunkSizes,'fisherPerClass')
        cfg.chunkSizes.fisherPerClass = [];
    end
end

