function spider_plot_R2019b(P, varargin)
%SPIDER_PLOT_R2019B Simple radar/spider plot compatible with Octave.
%   spider_plot_R2019b(P, 'AxesLabels', labels, 'AxesLimits', limits, ...)
%   creates a radar chart for the rows of P (data groups) across columns
%   (axes). This lightweight implementation supports a subset of the
%   options of the original FileExchange version and avoids MATLAB-specific
%   syntax so it runs in both Octave and MATLAB.
%
%   Supported name/value pairs:
%     'AxesLabels'      - cell array of axis labels
%     'AxesLimits'      - 2-by-N array of axis min/max values
%     'AxesInterval'    - number of intervals between min and max (default 3)
%     'AxesPrecision'   - number of decimals for tick labels (default 1)
%     'FillOption'      - 'on' or 'off' to fill polygons (default 'off')
%     'FillTransparency'- scalar or vector alpha value(s) for fill (default 0.1)
%     'Color'           - M-by-3 matrix of RGB colours (default lines)
%     'LineWidth'       - line width of polygons (default 1.5)
%     'Marker'          - marker spec or cell array (default 'o')
%     'MarkerSize'      - marker size (default 36)

    p = inputParser;
    p.addParameter('AxesLabels', {});
    p.addParameter('AxesLimits', []);
    p.addParameter('AxesInterval', 3);
    p.addParameter('AxesPrecision', 1);
    p.addParameter('FillOption', 'off');
    p.addParameter('FillTransparency', 0.1);
    p.addParameter('Color', []);
    p.addParameter('LineWidth', 1.5);
    p.addParameter('Marker', 'o');
    p.addParameter('MarkerSize', 36);
    p.parse(varargin{:});
    opt = p.Results;

    [numGroups, numAxes] = size(P);
    angles = linspace(0, 2*pi, numAxes + 1);

    if isempty(opt.AxesLimits)
        minVal = min(P(:));
        maxVal = max(P(:));
    else
        minVal = min(opt.AxesLimits(1,:));
        maxVal = max(opt.AxesLimits(2,:));
    end
    if maxVal == minVal
        maxVal = minVal + 1;
    end
    ticks = linspace(minVal, maxVal, opt.AxesInterval + 1);
    P_scaled = (P - minVal) / (maxVal - minVal);
    tick_scaled = (ticks - minVal) / (maxVal - minVal);

    if isempty(opt.Color)
        opt.Color = lines(numGroups);
    end
    if ischar(opt.Marker)
        opt.Marker = repmat({opt.Marker}, numGroups, 1);
    elseif iscell(opt.Marker) && numel(opt.Marker) == 1
        opt.Marker = repmat(opt.Marker, numGroups, 1);
    end
    if numel(opt.FillTransparency) == 1
        opt.FillTransparency = repmat(opt.FillTransparency, numGroups, 1);
    end

    hold on;
    % Radial grid
    for t = tick_scaled
        plot(t * cos(angles), t * sin(angles), 'Color', [0.8 0.8 0.8]);
    end
    % Axes
    for i = 1:numAxes
        plot([0 cos(angles(i))], [0 sin(angles(i))], 'Color', [0.8 0.8 0.8]);
        if ~isempty(opt.AxesLabels)
            text(1.05*cos(angles(i)), 1.05*sin(angles(i)), opt.AxesLabels{i}, ...
                'HorizontalAlignment','center','VerticalAlignment','middle');
        end
    end

    % Data polygons
    for g = 1:numGroups
        rho = P_scaled(g, :);
        rho(end+1) = rho(1);
        x = rho .* cos(angles);
        y = rho .* sin(angles);
        if strcmpi(opt.FillOption, 'on')
            patch(x, y, opt.Color(min(g,end),:), 'FaceAlpha', opt.FillTransparency(min(g,end)), ...
                'EdgeColor', 'none');
        end
        plot(x, y, 'Color', opt.Color(min(g,end),:), 'LineWidth', opt.LineWidth);
        plot(x, y, opt.Marker{min(g,end)}, 'Color', opt.Color(min(g,end),:), ...
            'MarkerFaceColor', opt.Color(min(g,end),:), 'MarkerSize', opt.MarkerSize);
    end

    axis equal off;
    % Tick labels along positive y-axis
    for k = 1:numel(ticks)
        text(0, tick_scaled(k), num2str(ticks(k), ['%0.' num2str(opt.AxesPrecision) 'f']), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom');
    end
    hold off;
end
