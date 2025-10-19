function colIdx = find_positive_class_column(classNames, positiveClassLabel)
%FIND_POSITIVE_CLASS_COLUMN Resolve the column index for the positive class.
%
%   COLIDX = FIND_POSITIVE_CLASS_COLUMN(CLASSNAMES, POSITIVECLASSLABEL)
%   returns the index in CLASSNAMES that corresponds to POSITIVECLASSLABEL.
%   If the positive class cannot be found the function returns empty.

    colIdx = [];
    if isnumeric(classNames)
        colIdx = find(classNames == positiveClassLabel, 1, 'first');
    elseif iscell(classNames)
        try
            numericNames = cellfun(@str2double, classNames);
            if all(~isnan(numericNames))
                colIdx = find(numericNames == positiveClassLabel, 1, 'first');
            end
        catch
            % Fallback to string comparison below
        end
        if isempty(colIdx)
            colIdx = find(strcmp(classNames, num2str(positiveClassLabel)), 1, 'first');
        end
    elseif isstring(classNames)
        numericNames = str2double(classNames);
        if all(~isnan(numericNames))
            colIdx = find(numericNames == positiveClassLabel, 1, 'first');
        end
        if isempty(colIdx)
            colIdx = find(classNames == string(positiveClassLabel), 1, 'first');
        end
    elseif iscategorical(classNames)
        try
            numericNames = double(classNames);
            if all(~isnan(numericNames))
                colIdx = find(numericNames == positiveClassLabel, 1, 'first');
            end
        catch
            % ignore
        end
        if isempty(colIdx)
            colIdx = find(classNames == categorical(positiveClassLabel), 1, 'first');
        end
    end
end
