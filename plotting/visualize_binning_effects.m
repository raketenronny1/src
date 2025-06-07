function visualize_binning_effects(P, opts)
%VISUALIZE_BINNING_EFFECTS Show the impact of different binning factors.
%
%   VISUALIZE_BINNING_EFFECTS(P, opts) loads the mean training spectrum and
%   plots how binning alters its appearance.
%   P   - project paths struct from SETUP_PROJECT_PATHS (optional)
%   opts- plotting options from PLOT_SETTINGS (optional)

    if nargin < 1 || isempty(P)
        P = setup_project_paths(pwd);
    end
    if nargin < 2 || isempty(opts)
        opts = plot_settings();
    end

    fprintf('Visualizing Binning Effects - %s\n', string(datetime('now')));

    projectRoot = P.projectRoot;
    dataPath    = P.dataPath;
    figuresPath = fullfile(P.figuresPath, 'SideQuests');
    if ~exist(figuresPath, 'dir'), mkdir(figuresPath); end
    dateStr = opts.datePrefix;

    %% Load Data
    trainingDataFile = fullfile(dataPath, 'training_set_no_outliers_T2Q.mat');
    loadedTrainingData = load(trainingDataFile, 'X_train_no_outliers');
    X_train_full = loadedTrainingData.X_train_no_outliers;
    wavenumbers_data = load(fullfile(dataPath, 'wavenumbers.mat'), 'wavenumbers_roi');
    wavenumbers_original = wavenumbers_data.wavenumbers_roi;
    mean_spectrum_original = mean(X_train_full, 1);

    binningFactorsToVisualize = [1, 2, 4, 8, 16];
    numPlots = numel(binningFactorsToVisualize);

    figure('Name', 'Effect of Spectral Binning', 'Position', [100, 100, 700, 900]);
    t = tiledlayout(numPlots,1,'TileSpacing','compact','Padding','compact');
    title(t, 'Auswirkung von Spektralem Binning auf ein Beispielspektrum', 'FontSize', opts.plotFontSize+2, 'FontWeight','bold');
    xlabel(t, opts.plotXLabel, 'FontSize', opts.plotFontSize);
    ylabel(t, opts.plotYLabelAbsorption, 'FontSize', opts.plotFontSize);

    for i = 1:numPlots
        binFactor = binningFactorsToVisualize(i);
        ax = nexttile;
        if binFactor == 1
            current_spectrum_to_plot = mean_spectrum_original;
            current_wavenumbers_to_plot = wavenumbers_original;
            plotTitle = 'Originalspektrum (Kein Binning, Faktor 1)';
        else
            [binned_spectrum, binned_wavenumbers] = bin_spectra(mean_spectrum_original, wavenumbers_original, binFactor);
            current_spectrum_to_plot = binned_spectrum;
            current_wavenumbers_to_plot = binned_wavenumbers;
            plotTitle = sprintf('Gebinnt mit Faktor %d (%d Merkmale)', binFactor, numel(binned_wavenumbers));
        end
        plot(ax, current_wavenumbers_to_plot, current_spectrum_to_plot, 'LineWidth', 1.5);
        title(ax, plotTitle, 'FontSize', opts.plotFontSize);
        grid(ax,'on'); ax.XDir = 'reverse';
        xlim(ax, [min(wavenumbers_original) max(wavenumbers_original)]);
        if i < numPlots, set(ax,'XTickLabel',[]); end
    end

    outBase = fullfile(figuresPath, sprintf('%s_SideQuest_BinningEffects', dateStr));
    savefig(gcf, [outBase '.fig']);
    exportgraphics(gcf, [outBase '.tiff'], 'Resolution',300);
    fprintf('Binning effect visualization plot saved to: %s.(fig/tiff)\n', outBase);
end
