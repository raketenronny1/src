function varargout = cached_pca(varargin)
% cached_pca Cached wrapper around MATLAB's PCA implementation.
%   This helper memoises PCA results keyed by the data matrix and an
%   optional hyperparameter/configuration signature. Subsequent calls with
%   identical data and configuration reuse the cached decomposition instead
%   of recomputing it.
%
%   Usage:
%     [coeff, score, latent, tsquared, explained, mu] = cached_pca(X);
%     [coeff, score, latent, tsquared, explained, mu] = cached_pca(X, cacheConfig, pcaArgs...);
%
%   Special Commands:
%     infoStruct = cached_pca('info');  % Retrieve cache statistics.
%     cached_pca('clear');              % Clear cached PCA results.
%
%   cacheConfig is optional and may contain the following fields:
%     - signature: arbitrary value that is incorporated into the cache key
%                  (e.g. hyperparameters controlling downstream behaviour).
%     - bypassCache: logical flag that forces recomputation when true.
%
%   Any additional positional arguments are forwarded to MATLAB's PCA
%   function as-is.

persistent pcaCache cacheStats

if isempty(pcaCache)
    pcaCache = containers.Map('KeyType','char','ValueType','any');
    cacheStats = initialise_stats();
end

if nargin == 0
    error('cached_pca:InvalidInput', 'Input data matrix X is required.');
end

firstArg = varargin{1};
if ischar(firstArg) || (isstring(firstArg) && isscalar(firstArg))
    command = lower(string(firstArg));
    switch command
        case "info"
            varargout{1} = build_info_struct(pcaCache, cacheStats);
            return;
        case "clear"
            pcaCache = containers.Map('KeyType','char','ValueType','any');
            cacheStats = initialise_stats();
            if nargout > 0
                varargout{1} = build_info_struct(pcaCache, cacheStats);
            end
            return;
        otherwise
            error('cached_pca:UnknownCommand', 'Unknown command "%s".', command);
    end
end

X = firstArg;
cacheConfig = struct();
idxNext = 2;
if nargin >= 2 && isstruct(varargin{2})
    cacheConfig = varargin{2};
    idxNext = 3;
end
pcaArgs = varargin(idxNext:end);

if ~isfield(cacheConfig, 'bypassCache') || ~cacheConfig.bypassCache
    cacheKey = build_cache_key(X, cacheConfig, pcaArgs);
else
    cacheKey = '';
end

useCache = ~isempty(cacheKey);
if useCache && pcaCache.isKey(cacheKey)
    cacheStats.hits = cacheStats.hits + 1;
    cachedEntry = pcaCache(cacheKey);
    cachedOutputs = cachedEntry.outputs;
    numStored = numel(cachedOutputs);
    numToReturn = min(nargout, numStored);
    if nargout > 0 && numToReturn > 0
        [varargout{1:numToReturn}] = cachedOutputs{1:numToReturn};
    end
    if nargout > numStored
        for k = numStored+1:nargout
            varargout{k} = [];
        end
    end
    return;
end

[coeff, score, latent, tsquared, explained, mu] = pca(X, pcaArgs{:});
outputs = {coeff, score, latent, tsquared, explained, mu};

if useCache
    cacheStats.misses = cacheStats.misses + 1;
    cacheEntry.outputs = outputs;
    cacheEntry.sizeBytes = estimate_struct_bytes(cacheEntry);
    cacheEntry.timestamp = datetime('now');
    cacheEntry.signature = cacheConfig_signature(cacheConfig);
    cacheEntry.dataSize = size(X);
    pcaCache(cacheKey) = cacheEntry;
    cacheStats.totalBytes = cacheStats.totalBytes + cacheEntry.sizeBytes;
end

[varargout{1:min(nargout, numel(outputs))}] = outputs{1:min(nargout, numel(outputs))};
if nargout > numel(outputs)
    for k = numel(outputs)+1:nargout
        varargout{k} = [];
    end
end

end

function stats = initialise_stats()
stats = struct('hits', 0, 'misses', 0, 'totalBytes', 0);
end

function info = build_info_struct(cacheMap, stats)
info = struct();
info.numEntries = cacheMap.Count;
info.hits = stats.hits;
info.misses = stats.misses;
info.totalBytes = stats.totalBytes;
info.totalMB = stats.totalBytes / (1024^2);
end

function cacheKey = build_cache_key(X, cacheConfig, pcaArgs)
configSignature = cacheConfig_signature(cacheConfig);
argsSignature = cacheConfig_signature(pcaArgs);
XHash = compute_data_hash(X);
cacheKey = sprintf('%s|%s|%s', XHash, configSignature, argsSignature);
end

function sig = cacheConfig_signature(data)
try
    sig = compute_data_hash(jsonencode(order_fields(data)));
catch
    sig = compute_data_hash(jsonencode(data));
end
end

function ordered = order_fields(input)
if isstruct(input)
    f = sort(fieldnames(input));
    ordered = struct();
    for i = 1:numel(f)
        fieldName = f{i};
        ordered.(fieldName) = order_fields(input.(fieldName));
    end
elseif iscell(input)
    ordered = cellfun(@order_fields, input, 'UniformOutput', false);
else
    ordered = input;
end
end

function hash = compute_data_hash(data)
engine = java.security.MessageDigest.getInstance('SHA-256');
metadata = struct('class', class(data), 'size', size(data));
metadataJson = jsonencode(metadata);
engine.update(uint8(metadataJson));
if isnumeric(data) || islogical(data)
    dataBytes = typecast(data(:), 'uint8');
    engine.update(dataBytes);
elseif ischar(data)
    engine.update(uint8(data(:)));
elseif isstring(data)
    engine.update(uint8(char(data)));
elseif iscell(data)
    for i = 1:numel(data)
        partialHash = compute_data_hash(data{i});
        engine.update(uint8(partialHash));
    end
elseif isstruct(data)
    fields = sort(fieldnames(data));
    for i = 1:numel(fields)
        fieldHash = compute_data_hash(fields{i});
        valueHash = compute_data_hash(data.(fields{i}));
        engine.update(uint8(fieldHash));
        engine.update(uint8(valueHash));
    end
elseif isa(data, 'datetime')
    engine.update(uint8(char(string(data))));
else
    encoded = getByteStreamFromArray(data);
    engine.update(uint8(encoded));
end
digest = uint8(engine.digest());
hash = lower(reshape(dec2hex(digest, 2).', 1, []));
end

function bytes = estimate_struct_bytes(s)
info = whos('s');
bytes = info.bytes;
end
