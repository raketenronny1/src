function info = get_parallel_environment_info()
%GET_PARALLEL_ENVIRONMENT_INFO Detect Parallel Computing Toolbox availability.
%   INFO = GET_PARALLEL_ENVIRONMENT_INFO() returns a struct describing the
%   current MATLAB parallel environment. Fields include:
%       toolboxDetected  - true if the Parallel Computing Toolbox is installed
%       licenseAvailable - true if a license for the toolbox is available
%       canUseParpool    - true if PARPOOL can be used (based on CANUSEPARPOOL)
%       isAvailable      - true when parallel features are fully usable
%       message          - descriptive status string
%
%   The helper caches the detection result for subsequent calls in order to
%   avoid repeated license checks.
%
%   Date: 2025-07-07

    persistent cachedInfo
    if ~isempty(cachedInfo)
        info = cachedInfo;
        return;
    end

    info = struct('toolboxDetected', false, ...
                  'licenseAvailable', false, ...
                  'canUseParpool', false, ...
                  'isAvailable', false, ...
                  'message', '');

    % Detect installation by querying the toolbox table via VER.
    try
        info.toolboxDetected = ~isempty(ver('parallel'));
    catch
        info.toolboxDetected = false;
    end

    if ~info.toolboxDetected
        info.message = 'Parallel Computing Toolbox not installed.';
        cachedInfo = info;
        return;
    end

    % Check whether the license is present.
    try
        info.licenseAvailable = license('test','Distrib_Computing_Toolbox');
    catch
        info.licenseAvailable = false;
    end

    if ~info.licenseAvailable
        info.message = 'Parallel Computing Toolbox license unavailable.';
        cachedInfo = info;
        return;
    end

    % Query CANUSEPARPOOL when available to detect cluster restrictions.
    if exist('canUseParpool','file') == 2
        try
            info.canUseParpool = canUseParpool();
        catch
            info.canUseParpool = false;
        end
    else
        % Older MATLAB releases may not implement CANUSEPARPOOL; assume parpool
        % can be used when the toolbox and license are present.
        info.canUseParpool = true;
    end

    if info.canUseParpool
        info.isAvailable = true;
        info.message = 'Parallel Computing Toolbox available.';
    else
        info.message = 'Parallel pool unavailable in current configuration.';
    end

    cachedInfo = info;
end
