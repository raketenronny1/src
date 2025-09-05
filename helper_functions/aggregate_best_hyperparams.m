function agg = aggregate_best_hyperparams(hyperStructs)
%AGGREGATE_BEST_HYPERPARAMS Combine hyperparameter selections across folds.
%   AGG = AGGREGATE_BEST_HYPERPARAMS(HC) takes a cell array HC where each
%   element is a struct of hyperparameters chosen on an inner fold. The
%   function returns a struct AGG in which each field value is the mode of
%   the corresponding values across folds (ignoring missing entries).
%
%   This helper provides a simple way to derive a single set of
%   hyperparameters from the per‑fold selections produced during nested
%   cross‑validation.
%
%   Date: 2025-06-16

    agg = struct();
    if isempty(hyperStructs)
        return;
    end

    % Collect all field names appearing in the structs
    allFields = {};
    for i = 1:numel(hyperStructs)
        if ~isempty(hyperStructs{i})
            allFields = union(allFields, fieldnames(hyperStructs{i}));
        end
    end

    for f = 1:numel(allFields)
        fld = allFields{f};
        vals = [];
        for i = 1:numel(hyperStructs)
            hs = hyperStructs{i};
            if isstruct(hs) && isfield(hs, fld)
                vals = [vals, hs.(fld)]; %#ok<AGROW>
            end
        end
        if isempty(vals)
            continue;
        end
        if isnumeric(vals)
            aggVal = mode(vals);
        else
            % Convert to string for mode calculation
            strVals = string(vals);
            [u,~,ic] = unique(strVals);
            counts = accumarray(ic,1);
            [~,idx] = max(counts);
            aggVal = u(idx);
            if ischar(vals{1})
                aggVal = char(aggVal);
            end
        end
        agg.(fld) = aggVal;
    end
end
