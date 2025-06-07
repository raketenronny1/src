function opts = plot_settings()
%PLOT_SETTINGS  Return global plotting options used across visualization functions.
%
%   opts = PLOT_SETTINGS() returns a struct with colour codes and
%   formatting defaults for the project plots.
%
%   Date: 2025-06-07

    opts.colorWHO1 = [0.9, 0.6, 0.4];
    opts.colorWHO3 = [0.4, 0.702, 0.902];
    opts.colorT2OutlierFlag = [0.8, 0.2, 0.2];
    opts.colorQOutlierFlag  = [0.2, 0.2, 0.8];
    opts.colorBothOutlierFlag = [0.8, 0, 0.8];
    opts.colorCV   = [0.2, 0.6, 0.2];
    opts.colorTest = [0.8, 0.2, 0.2];

    opts.plotFontSize = 12;
    opts.plotXLabel   = 'Wellenzahl (cm^{-1})';
    opts.plotYLabelAbsorption = 'Absorption (a.u.)';
    opts.plotXLim = [950 1800];

    opts.datePrefix = string(datetime('now','Format','yyyyMMdd'));
end
