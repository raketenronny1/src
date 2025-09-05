function visualize_spectral_heatmap(P, opts)
%VISUALIZE_SPECTRAL_HEATMAP Display heatmaps of preprocessed spectra grouped by class.
%
%   VISUALIZE_SPECTRAL_HEATMAP(P, opts) loads the table produced by
%   RUN_SPECTRAL_PREPROCESSING_WORKFLOW (containing a column
%   "FinalProcessedSpectrum") and plots heatmaps of all spectra grouped by
%   their class label.
%   P   - project paths struct from SETUP_PROJECT_PATHS (optional)
%   opts- plotting options from PLOT_SETTINGS (optional)
%
%   The function expects a MAT-file named
%   "spectral_preprocessing_output.mat" in the data directory that contains
%   variables "data_all_positions" and "wavenumbers_roi".

    if nargin < 1 || isempty(P)
        P = setup_project_paths(pwd);
    end
    if nargin < 2 || isempty(opts)
        opts = plot_settings();
    end

    fprintf('Generating spectral heatmaps - %s\n', string(datetime('now')));

    dataPath    = P.dataPath;
    figuresPath = fullfile(P.figuresPath, 'SpectralHeatmap');
    if ~exist(figuresPath, 'dir'), mkdir(figuresPath); end
    dateStr = opts.datePrefix;

    %% Load preprocessed spectra
    dataFile = fullfile(dataPath, 'spectral_preprocessing_output.mat');
    if ~isfile(dataFile)
        error('Preprocessed spectra file not found: %s. Run run_spectral_preprocessing_workflow first.', dataFile);
    end
    S = load(dataFile, 'data_all_positions', 'wavenumbers_roi');
    if ~isfield(S, 'data_all_positions') || ~isfield(S, 'wavenumbers_roi')
        error('Required variables missing in %s. Expected data_all_positions and wavenumbers_roi.', dataFile);
    end
    data_all_positions = S.data_all_positions;
    wavenumbers_roi = S.wavenumbers_roi;
    if iscolumn(wavenumbers_roi), wavenumbers_roi = wavenumbers_roi'; end

    if ~ismember('FinalProcessedSpectrum', data_all_positions.Properties.VariableNames)
        error('data_all_positions lacks "FinalProcessedSpectrum" column.');
    end

    %% Determine class labels
    classField = '';
    if ismember('WHO_Grade', data_all_positions.Properties.VariableNames)
        classField = 'WHO_Grade';
    elseif ismember('WHO_Grade_str', data_all_positions.Properties.VariableNames)
        classField = 'WHO_Grade_str';
    elseif ismember('classLabel', data_all_positions.Properties.VariableNames)
        classField = 'classLabel';
    else
        error('No class label column found in data_all_positions.');
    end
    class_labels = data_all_positions.(classField);
    class_labels = cellstr(string(class_labels));

    unique_classes = unique(class_labels);
    numClasses = numel(unique_classes);

    %% Assemble figure
    fig = figure('Name', 'Spectral Heatmaps');
    t = tiledlayout(numClasses,1,'TileSpacing','compact','Padding','compact');
    title(t, 'Spectral Heatmaps by Class', 'FontSize', opts.plotFontSize+2, 'FontWeight','bold');
    xlabel(t, opts.plotXLabel, 'FontSize', opts.plotFontSize);

    for i = 1:numClasses
        thisClass = unique_classes{i};
        idx = strcmp(class_labels, thisClass);
        spectraBlocks = data_all_positions.FinalProcessedSpectrum(idx);
        spectraMat = vertcat(spectraBlocks{:});

        ax = nexttile;
        if isempty(spectraMat)
            text(0.5,0.5,'No spectra','Parent',ax,'HorizontalAlignment','center');
        else
            imagesc(ax, wavenumbers_roi, 1:size(spectraMat,1), spectraMat);
            set(ax,'YDir','normal','XDir','reverse');
            ylabel(ax, sprintf('Index (n=%d)', size(spectraMat,1)), 'FontSize', opts.plotFontSize);
        end
        title(ax, sprintf('%s', thisClass), 'Interpreter','none', 'FontSize', opts.plotFontSize);
        colorbar(ax);
    end
    colormap(t, parula);

    outBase = fullfile(figuresPath, sprintf('%s_SpectralHeatmap', dateStr));
    savefig(fig, [outBase '.fig']);
    exportgraphics(fig, [outBase '.tiff'], 'Resolution', 300);
    fprintf('Spectral heatmap saved to: %s.(fig/tiff)\n', outBase);
end
