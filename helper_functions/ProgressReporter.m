classdef ProgressReporter < handle
    %PROGRESSREPORTER Lightweight console progress updates with TTY detection.
    %   reporter = ProgressReporter(label, totalSteps) creates a progress
    %   reporter that tracks the number of completed steps. Call update() to
    %   advance the counter and emit a console message. By default the class
    %   attempts to update the same line in interactive MATLAB sessions and
    %   prints discrete lines in headless environments.
    %
    %   Optional name/value pairs:
    %       'Verbose'   - logical flag (default true). When false, update() is
    %                    a no-op.
    %       'ThrottleSeconds' - minimum seconds between console refreshes
    %                    (default 0.2). Set to 0 to disable throttling.
    %
    %   Example:
    %       pr = ProgressReporter('Outer CV', 5);
    %       for k = 1:5
    %           pause(0.1);
    %           pr.update(1, sprintf('Fold %d complete', k));
    %       end
    %
    %   The reporter automatically prints a trailing newline when the total
    %   number of steps is reached.
    %
    %   Date: 2025-06-09

    properties (Access = private)
        Label (1,1) string = "Progress"
        TotalSteps (1,1) double = 1
        CompletedSteps (1,1) double = 0
        Verbose (1,1) logical = true
        IsInteractive (1,1) logical = true
        LastMessageLength (1,1) double = 0
        LastUpdateTime (1,1) double = -Inf
        ThrottleSeconds (1,1) double = 0.2
    end

    methods
        function obj = ProgressReporter(label, totalSteps, varargin)
            if nargin < 1 || isempty(label)
                label = "Progress";
            end
            if nargin < 2 || isempty(totalSteps)
                totalSteps = 1;
            end

            obj.Label = string(label);
            obj.TotalSteps = max(1, double(totalSteps));
            obj.IsInteractive = ProgressReporter.detectInteractiveSession();

            if ~isempty(varargin)
                p = inputParser();
                addParameter(p, 'Verbose', true, @(v) islogical(v) || isnumeric(v));
                addParameter(p, 'ThrottleSeconds', 0.2, @(v) isnumeric(v) && isscalar(v) && v >= 0);
                parse(p, varargin{:});
                obj.Verbose = logical(p.Results.Verbose);
                obj.ThrottleSeconds = double(p.Results.ThrottleSeconds);
            end

            if ~obj.Verbose
                obj.ThrottleSeconds = 0;
            end
        end

        function update(obj, stepIncrement, message)
            %UPDATE Advance the progress counter and optionally print a message.
            %   UPDATE(reporter) increments by 1.
            %   UPDATE(reporter, stepIncrement) increments by stepIncrement.
            %   UPDATE(reporter, stepIncrement, message) also appends a
            %   user-provided message to the console output.

            if ~obj.Verbose
                return;
            end

            if nargin < 2 || isempty(stepIncrement)
                stepIncrement = 1;
            end
            if nargin < 3
                message = "";
            end

            stepIncrement = double(stepIncrement);
            if ~isscalar(stepIncrement) || ~isfinite(stepIncrement)
                error('ProgressReporter:InvalidStep', 'Step increment must be a finite scalar.');
            end

            obj.CompletedSteps = max(0, min(obj.TotalSteps, obj.CompletedSteps + stepIncrement));
            pct = (obj.CompletedSteps / obj.TotalSteps) * 100;

            if strlength(string(message)) > 0
                msgSuffix = " - " + string(message);
            else
                msgSuffix = "";
            end

            baseMsg = sprintf('%s: %d/%d (%.1f%%%%)', obj.Label, ...
                round(obj.CompletedSteps), round(obj.TotalSteps), pct);
            fullMsg = string(baseMsg) + msgSuffix;

            if obj.ThrottleSeconds > 0
                currentTime = ProgressReporter.monotonicSeconds();
                if isfinite(obj.LastUpdateTime) && currentTime - obj.LastUpdateTime < obj.ThrottleSeconds ...
                        && obj.CompletedSteps < obj.TotalSteps
                    return;
                end
                obj.LastUpdateTime = currentTime;
            end

            if obj.IsInteractive
                obj.printInteractive(fullMsg);
            else
                obj.printNonInteractive(fullMsg);
            end
        end
    end

    methods (Access = private)
        function printInteractive(obj, fullMsg)
            %PRINTINTERACTIVE Use carriage return to refresh the console line.
            msgChar = char(fullMsg);
            msgLength = numel(msgChar);
            padding = max(0, obj.LastMessageLength - msgLength);
            fprintf('\r%s%s', msgChar, repmat(' ', 1, padding));
            obj.LastMessageLength = msgLength;
            if obj.CompletedSteps >= obj.TotalSteps
                fprintf('\n');
                obj.LastMessageLength = 0;
            end
        end

        function printNonInteractive(obj, fullMsg)
            %PRINTNONINTERACTIVE Emit a full line for each update to avoid
            % carriage-return artefacts in headless logs.
            fprintf('%s\n', char(fullMsg));
        end
    end

    methods (Static, Access = private)
        function tf = detectInteractiveSession()
            %DETECTINTERACTIVESESSION Best-effort detection of an interactive console.
            tf = false;
            try
                tf = usejava('desktop') && feature('hotlinks');
            catch
                tf = false;
            end
        end

        function t = monotonicSeconds()
            %MONOTONICSECONDS Return seconds since MATLAB start using tic/toc.
            persistent tStart;
            if isempty(tStart)
                tStart = tic;
            end
            t = toc(tStart);
        end
    end
end
