% run_visualize_phase3_results.m
% Script specifically for visualizing Phase 3 results based on provided file information.

% ------------------------------------------------------------------------
% 1. Initialization & Configuration
% ------------------------------------------------------------------------
clearvars; close all; clc;
fprintf('============================================================\n');
fprintf('Starting Visualization of Phase 3 Results (%s)\n', datestr(now));
fprintf('============================================================\n\n');

% --- Define PATH to your results ---
% !!! USER ACTION: Make sure this path is correct !!!
resultsPath = 'C:\Users\Franz\OneDrive\01_Promotion\01 Data\meningioma-ftir-classification\results\Phase3';
fprintf('Results Path: %s\n', resultsPath);

% --- Define EXACT filenames for Phase 3 results ---
% Option 1: Visualize results from the "Stratified OR" strategy
spectrumLevelResultsFile = fullfile(resultsPath, '20250518_Phase3_TestSetResults_Strat_OR.mat');
probeLevelResultsFile    = fullfile(resultsPath, '20250518_Phase3_TestSetResults_Strat_OR.mat'); % This file also contains probe results

% Option 2: Visualize results from the other "TestSetResults" file
% spectrumLevelResultsFile = fullfile(resultsPath, '20250515_Phase3_TestSetResults.mat');
% For this option, the probe-level data would come from its dedicated file:
% probeLevelResultsFile    = fullfile(resultsPath, '20250515_Phase3_ProbeLevelTestSetResults.mat');

fprintf('Using Spectrum-Level Results File: %s\n', spectrumLevelResultsFile);
fprintf('Using Probe-Level Results File: %s\n\n', probeLevelResultsFile);

% --- Define class information ---
classNames         = {'WHO I', 'WHO III'};
positiveClassName  = 'WHO III'; % For plot labels
positiveClassLabel = 1;         % Numeric label for the positive class (e.g., if WHO I is 0/false, WHO III is 1/true)
% You might need to confirm how your labels are numerically encoded (e.g., 0/1, 1/2, or logical true/false)

% --- Plotting Toggles ---
doPlotConfusionMatrix_Spectrum   = true;
doPlotRocDet_Spectrum            = true;
doPlotProbabilityDistribution_Probe = true;

% ------------------------------------------------------------------------
% 2. Data Loading
% ------------------------------------------------------------------------
fprintf('--- Loading Data ---\n');

% --- Load Spectrum-Level Test Set Results ---
loadedSpectrumData = []; % Initialize
if doPlotConfusionMatrix_Spectrum || doPlotRocDet_Spectrum
    try
        fprintf('Attempting to load spectrum-level results from: %s\n', spectrumLevelResultsFile);
        if exist(spectrumLevelResultsFile, 'file')
            loadedSpectrumData = load(spectrumLevelResultsFile);
            fprintf('Successfully loaded spectrum-level results.\n');
            % Display contents to help user identify variable names:
            disp('Variables in loadedSpectrumData:');
            disp(loadedSpectrumData);
            
            % Expected to contain: testSetPerformanceMetrics_spectrum (if using _Strat_OR.mat)
            % or testSetPerformanceMetrics (if using 20250515_Phase3_TestSetResults.mat)
            if isfield(loadedSpectrumData, 'testSetPerformanceMetrics_spectrum')
                fprintf('Found ''testSetPerformanceMetrics_spectrum''.\n');
            elseif isfield(loadedSpectrumData, 'testSetPerformanceMetrics')
                fprintf('Found ''testSetPerformanceMetrics''.\n');
            else
                warning('Could not find ''testSetPerformanceMetrics_spectrum'' or ''testSetPerformanceMetrics'' in the loaded spectrum-level data.');
                loadedSpectrumData = []; % Mark as empty if key data is missing
            end
        else
            warning('Spectrum-level results file not found: %s', spectrumLevelResultsFile);
        end
    catch ME
        warning('Error loading spectrum-level results from %s: %s', spectrumLevelResultsFile, ME.message);
        loadedSpectrumData = [];
    end
else
    fprintf('Skipping loading of spectrum-level results data as plots are disabled.\n');
end
fprintf('\n');

% --- Load Probe-Level Test Set Results ---
loadedProbeData = []; % Initialize
if doPlotProbabilityDistribution_Probe
    try
        fprintf('Attempting to load probe-level results from: %s\n', probeLevelResultsFile);
        if exist(probeLevelResultsFile, 'file')
            loadedProbeData = load(probeLevelResultsFile);
            fprintf('Successfully loaded probe-level results.\n');
            % Display contents:
            disp('Variables in loadedProbeData:');
            disp(loadedProbeData);
            
            % Expected to contain: probeLevelResults (table)
            if isfield(loadedProbeData, 'probeLevelResults') && istable(loadedProbeData.probeLevelResults)
                fprintf('Found ''probeLevelResults'' table.\n');
                disp('Column names in probeLevelResults table:');
                disp(loadedProbeData.probeLevelResults.Properties.VariableNames);
            else
                warning('Could not find ''probeLevelResults'' table in the loaded probe-level data.');
                loadedProbeData = []; % Mark as empty
            end
        else
            warning('Probe-level results file not found: %s', probeLevelResultsFile);
        end
    catch ME
        warning('Error loading probe-level results from %s: %s', probeLevelResultsFile, ME.message);
        loadedProbeData = [];
    end
else
    fprintf('Skipping loading of probe-level results data as plot is disabled.\n');
end
fprintf('\n');


% ------------------------------------------------------------------------
% 3. Data Extraction & Plotting Calls
%    !!! USER ACTION: You will LIKELY need to adjust the lines below
%    where variables like 'trueTestLabels_spectrum', 'scores_spectrum',
%    'probeTrueLabels', 'probeMeanProbabilities' are extracted,
%    based on the actual structure and field/column names in your loaded data.
% ------------------------------------------------------------------------
fprintf('--- Generating Visualizations for Phase 3 ---\n');

% --- 3.1 Confusion Matrix (Spectrum-Level) ---
if doPlotConfusionMatrix_Spectrum && ~isempty(loadedSpectrumData)
    fprintf('Plotting Spectrum-Level Confusion Matrix...\n');
    try
        % *** ASSUMPTION & USER ACTION REQUIRED ***
        % Access the correct performance metrics struct:
        if isfield(loadedSpectrumData, 'testSetPerformanceMetrics_spectrum')
            metricsStruct = loadedSpectrumData.testSetPerformanceMetrics_spectrum;
        elseif isfield(loadedSpectrumData, 'testSetPerformanceMetrics')
            metricsStruct = loadedSpectrumData.testSetPerformanceMetrics;
        else
            error('Performance metrics struct not found in loadedSpectrumData.');
        end
        
        % *** ASSUMPTION & USER ACTION REQUIRED ***
        % Extract true labels, predicted labels. These names are GUESSES.
        % You need to look inside your 'metricsStruct' (e.g., by setting a breakpoint and inspecting it)
        % or know how your 'calculate_performance_metrics.m' (or similar) function stores these.
        % Common names might be: 'TrueLabels', 'PredictedLabels', 'YTrue', 'YPred', 'actual', 'predicted'
        % Assuming they are stored as, for example: metricsStruct.TrueLabels and metricsStruct.PredictedLabels
        
        if isfield(metricsStruct, 'TrueLabels') && isfield(metricsStruct, 'PredictedLabels')
            trueTestLabels_spectrum    = metricsStruct.TrueLabels;
            predictedTestLabels_spectrum = metricsStruct.PredictedLabels;

            % Ensure labels are categorical for confusionchart
            % Assuming numeric labels [0,1] or [negativeClassLabel, positiveClassLabel]
            trueL_cat = categorical(trueTestLabels_spectrum, [negativeClassLabel, positiveClassLabel], classNames);
            predL_cat = categorical(predictedTestLabels_spectrum, [negativeClassLabel, positiveClassLabel], classNames);

            plotConfusionMatrix(trueL_cat, predL_cat, classNames, ...
                'Confusion Matrix: Phase 3 Test Set (Spectrum-Level)');
        else
            warning('Could not find TrueLabels/PredictedLabels fields in the spectrum performance metrics struct. Skipping Confusion Matrix.');
        end
    catch ME
        warning('Failed to generate Spectrum-Level Confusion Matrix: %s', ME.message);
    end
    fprintf('\n');
else
    fprintf('Skipping Spectrum-Level Confusion Matrix (data missing or plot disabled).\n\n');
end

% --- 3.2 ROC and DET Curves (Spectrum-Level) ---
if doPlotRocDet_Spectrum && ~isempty(loadedSpectrumData)
    fprintf('Plotting Spectrum-Level ROC/DET Curves...\n');
    try
        % *** ASSUMPTION & USER ACTION REQUIRED *** (Same as above for metricsStruct)
        if isfield(loadedSpectrumData, 'testSetPerformanceMetrics_spectrum')
            metricsStruct = loadedSpectrumData.testSetPerformanceMetrics_spectrum;
        elseif isfield(loadedSpectrumData, 'testSetPerformanceMetrics')
            metricsStruct = loadedSpectrumData.testSetPerformanceMetrics;
        else
            error('Performance metrics struct not found for ROC/DET.');
        end

        % *** ASSUMPTION & USER ACTION REQUIRED ***
        % Extract true labels and scores for the positive class. These names are GUESSES.
        % Scores might be named: 'Scores', 'Probabilities', 'YScores_PositiveClass'
        % TrueLabels might be: 'TrueLabels', 'YTrue'
        % Ensure 'scores_spectrum' is a vector of probabilities for the POSITIVE class.
        if isfield(metricsStruct, 'TrueLabels') && isfield(metricsStruct, 'Scores_PositiveClass') % GUESS for scores field name
            trueTestLabels_spectrum = metricsStruct.TrueLabels;
            scores_spectrum         = metricsStruct.Scores_PositiveClass; % This needs to be scores for the positive class

            % Ensure true labels for perfcurve are numeric or logical if positiveClassLabel is numeric
            plotRocAndDetCurves(trueTestLabels_spectrum, ...
                                scores_spectrum, ...
                                positiveClassLabel, ... 
                                'ROC: Phase 3 Test Set (Spectrum-Level)', ...
                                'DET: Phase 3 Test Set (Spectrum-Level)', ...
                                true); % true for normal deviate scale on DET
        else
            warning('Could not find TrueLabels/Scores_PositiveClass fields in the spectrum performance metrics struct. Skipping ROC/DET.');
        end
    catch ME
        warning('Failed to generate Spectrum-Level ROC/DET Curves: %s', ME.message);
    end
    fprintf('\n');
else
    fprintf('Skipping Spectrum-Level ROC/DET Curves (data missing or plot disabled).\n\n');
end

% --- 3.3 Probability Distribution Plot (Probe-Level) ---
if doPlotProbabilityDistribution_Probe && ~isempty(loadedProbeData) && isfield(loadedProbeData, 'probeLevelResults')
    fprintf('Plotting Probe-Level Probability Distribution...\n');
    try
        probeTable = loadedProbeData.probeLevelResults; % This is a table

        % *** ASSUMPTION & USER ACTION REQUIRED ***
        % Identify the correct column names in your 'probeTable'.
        % Use `disp(probeTable.Properties.VariableNames);` to see them.
        % True labels column might be: 'TrueLabel', 'ActualLabel', 'WHO_Grade'
        % Probabilities column might be: 'MeanProbability_WHO_III', 'AggregatedScore_Positive', 'PredictionScore'
        
        % Example (GUESSES - you MUST verify these column names from your table):
        trueLabelColName = 'TrueLabel';         % <<< VERIFY THIS COLUMN NAME
        probPosClassColName = 'MeanScore_PosClass'; % <<< VERIFY THIS COLUMN NAME (Prob for WHO III)

        if ismember(trueLabelColName, probeTable.Properties.VariableNames) && ...
           ismember(probPosClassColName, probeTable.Properties.VariableNames)
            
            probeTrueLabels = probeTable.(trueLabelColName);
            % Ensure probeTrueLabels are in a format compatible with grouping (e.g., categorical or numeric)
            % If they are strings like 'WHO I', 'WHO III', they should work directly with boxplot.
            % If numeric (0,1), that's also fine.
            
            probeMeanProbabilities = probeTable.(probPosClassColName);

            plotProbabilityDistribution(probeMeanProbabilities, ...
                                        probeTrueLabels, ...
                                        positiveClassName, ... % For y-axis label text
                                        'Probe Probabilities (Boxplot) - Phase 3 Test Set', ...
                                        'Probe Probabilities (Dotplot) - Phase 3 Test Set');
        else
            warning('Required columns (''%s'' and/or ''%s'') not found in probeLevelResults table. Skipping Probability Distribution Plot.', trueLabelColName, probPosClassColName);
            disp('Available columns in probeLevelResults table are:');
            disp(probeTable.Properties.VariableNames);
        end
    catch ME
        warning('Failed to generate Probe-Level Probability Distribution Plot: %s', ME.message);
    end
    fprintf('\n');
else
    fprintf('Skipping Probe-Level Probability Distribution Plot (data missing or plot disabled).\n\n');
end


fprintf('============================================================\n');
fprintf('Phase 3 Visualization script finished (%s)\n', datestr(now));
fprintf('============================================================\n');


% ------------------------------------------------------------------------
% Helper Function Definitions (Copied from previous response)
% ------------------------------------------------------------------------
% function fh_cm = plotConfusionMatrix(...) ... end
% function fh_roc_det = plotRocAndDetCurves(...) ... end
% function fh_prob_dist = plotProbabilityDistribution(...) ... end
% (Paste the full function definitions here)

% --- PASTE HELPER FUNCTIONS HERE ---
function fh_cm = plotConfusionMatrix(trueLabels, predictedLabels, classNamesArg, titleStr)
% plotConfusionMatrix - Displays a confusion matrix.
    if nargin < 4 || isempty(titleStr)
        titleStr = 'Confusion Matrix';
    end

    fh_cm = figure('Name', titleStr, 'NumberTitle', 'off');
    try
        cm = confusionchart(trueLabels, predictedLabels);
        cm.Title = titleStr;
        if nargin >= 3 && ~isempty(classNamesArg)
            cm.ClassNames = classNamesArg;
        end
        % cm.ColumnSummary = 'column-normalized'; % Optional: Precision
        % cm.RowSummary = 'row-normalized';     % Optional: Recall
        fprintf('Confusion matrix ''%s'' plotted.\n', titleStr);
    catch ME
        fprintf('Error plotting confusion matrix: %s\n', ME.message);
        if ishandle(fh_cm); close(fh_cm); fh_cm = []; end % Close figure if error
    end
end

function fh_roc_det = plotRocAndDetCurves(trueLabels, scores, positiveClassIdentifier, titleRocStr, titleDetStr, useNormalDeviateScaleDet)
% plotRocAndDetCurves - Plots ROC and DET curves side-by-side.
    if nargin < 4 || isempty(titleRocStr)
        titleRocStr = 'ROC Curve';
    end
    if nargin < 5 || isempty(titleDetStr)
        titleDetStr = 'DET Curve';
    end
    if nargin < 6 || isempty(useNormalDeviateScaleDet)
        useNormalDeviateScaleDet = false;
    end

    fh_roc_det = figure('Name', [titleRocStr, ' & ', titleDetStr], 'NumberTitle', 'off');
    try
        [Xroc, Yroc, ~, AUCroc] = perfcurve(trueLabels, scores, positiveClassIdentifier);
        [Xdet_fpr, Ydet_fnr] = perfcurve(trueLabels, scores, positiveClassIdentifier, 'XCrit', 'fpr', 'YCrit', 'fnr');

        tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        ax_roc = nexttile;
        plot(ax_roc, Xroc, Yroc, 'LineWidth', 1.5);
        hold(ax_roc, 'on'); plot(ax_roc, [0 1], [0 1], 'k--'); hold(ax_roc, 'off');
        xlabel(ax_roc, 'False Positive Rate (FPR)'); ylabel(ax_roc, 'True Positive Rate (TPR)');
        title(ax_roc, sprintf('%s (AUC = %.3f)', titleRocStr, AUCroc));
        grid(ax_roc, 'on'); axis(ax_roc, [0 1 0 1]);

        ax_det = nexttile;
        if useNormalDeviateScaleDet
            epsilon = 1e-7; % Small value to avoid Inf with norminv
            % Ensure inputs to norminv are within (epsilon, 1-epsilon)
            Xdet_fpr_safe = max(epsilon, min(1-epsilon, Xdet_fpr));
            Ydet_fnr_safe = max(epsilon, min(1-epsilon, Ydet_fnr));

            Xdet_fpr_norm = norminv(Xdet_fpr_safe);
            Ydet_fnr_norm = norminv(Ydet_fnr_safe);
            
            plot(ax_det, Xdet_fpr_norm, Ydet_fnr_norm, 'LineWidth', 1.5);
            xlabel(ax_det, 'FPR (Normal Deviate Scale)'); ylabel(ax_det, 'FNR (Normal Deviate Scale)');
            title(ax_det, [titleDetStr, ' (Normal Deviate)']);
            
            prob_ticks = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999];
            norm_ticks = norminv(prob_ticks);
            
            % Filter ticks to be within the range of plotted data to avoid clutter
            x_ticks_to_use = norm_ticks(norm_ticks >= min(Xdet_fpr_norm(isfinite(Xdet_fpr_norm))) & norm_ticks <= max(Xdet_fpr_norm(isfinite(Xdet_fpr_norm))));
            y_ticks_to_use = norm_ticks(norm_ticks >= min(Ydet_fnr_norm(isfinite(Ydet_fnr_norm))) & norm_ticks <= max(Ydet_fnr_norm(isfinite(Ydet_fnr_norm))));

            if ~isempty(x_ticks_to_use)
                set(ax_det, 'XTick', x_ticks_to_use);
                xticklabels(ax_det, sprintfc('%.3g', prob_ticks(ismembertol(norm_ticks, x_ticks_to_use))));
            end
            if ~isempty(y_ticks_to_use)
                 set(ax_det, 'YTick', y_ticks_to_use);
                 yticklabels(ax_det, sprintfc('%.3g', prob_ticks(ismembertol(norm_ticks, y_ticks_to_use))));
            end

        else
            plot(ax_det, Xdet_fpr, Ydet_fnr, 'LineWidth', 1.5);
            xlabel(ax_det, 'False Positive Rate (FPR)'); ylabel(ax_det, 'False Negative Rate (FNR)');
            title(ax_det, titleDetStr); axis(ax_det, [0 1 0 1]);
        end
        grid(ax_det, 'on');
        fprintf('ROC and DET curves ''%s'' & ''%s'' plotted.\n', titleRocStr, titleDetStr);
    catch ME
        fprintf('Error plotting ROC/DET curves: %s\n', ME.message);
        if ishandle(fh_roc_det); close(fh_roc_det); fh_roc_det = []; end
    end
end

function fh_prob_dist = plotProbabilityDistribution(probeLevelProbabilities, trueProbeClasses, positiveClassNameText, titleBoxStr, titleDotStr)
% plotProbabilityDistribution - Displays boxplot and dotplot of probe-level probabilities.
    if nargin < 4 || isempty(titleBoxStr)
        titleBoxStr = 'Boxplot of Probe Probabilities';
    end
    if nargin < 5 || isempty(titleDotStr)
        titleDotStr = 'Dotplot of Probe Probabilities';
    end

    fh_prob_dist = figure('Name', [titleBoxStr, ' & ', titleDotStr], 'NumberTitle', 'off');
    try
        yLabelStr = sprintf('Aggregated Probability for %s', string(positiveClassNameText));
        tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        ax_box = nexttile;
        % Convert trueProbeClasses to categorical if it's not, ensuring correct labels for boxplot
        if isnumeric(trueProbeClasses) || islogical(trueProbeClasses)
            globalClassNames = {'WHO I', 'WHO III'}; % Assuming these are the only two
            globalNegativeLabel = 0; % Assuming WHO I is 0
            globalPositiveLabel = 1; % Assuming WHO III is 1
            displayProbeClasses = categorical(trueProbeClasses, [globalNegativeLabel, globalPositiveLabel], globalClassNames, 'Ordinal',false);
        elseif iscategorical(trueProbeClasses)
            displayProbeClasses = trueProbeClasses;
        else % try to convert strings/cellstrs
            displayProbeClasses = categorical(trueProbeClasses);
        end

        boxplot(ax_box, probeLevelProbabilities, displayProbeClasses, 'Whisker', 1.5);
        ylabel(ax_box, yLabelStr); title(ax_box, titleBoxStr);
        grid(ax_box, 'on'); ylim(ax_box, [0 1]);

        ax_dot = nexttile;
        classCats = categories(displayProbeClasses); % Get categories from the converted version
        if isempty(classCats) % if not categorical or single category
             classCats = unique(displayProbeClasses); % Fallback
        end

        numCats = length(classCats);
        colors = lines(numCats);

        for i = 1:numCats
            currentCat = classCats{i};
            isCurrentClass = (displayProbeClasses == currentCat);
            
            y_values = probeLevelProbabilities(isCurrentClass);
            if isempty(y_values); continue; end % Skip if no data for this category

            x_position = i;
            jitter_amount = 0.15;
            x_jittered = x_position + jitter_amount * (rand(size(y_values)) - 0.5);
            
            scatter(ax_dot, x_jittered, y_values, 36, 'filled', 'MarkerFaceColor', colors(i,:), 'MarkerFaceAlpha', 0.6);
            hold(ax_dot, 'on');
        end
        hold(ax_dot, 'off');
        
        if numCats > 0
            xticks(ax_dot, 1:numCats); 
            xticklabels(ax_dot, classCats);
        end
        ylabel(ax_dot, yLabelStr); title(ax_dot, titleDotStr);
        grid(ax_dot, 'on'); ylim(ax_dot, [0 1]);
        fprintf('Probability distribution plots ''%s'' & ''%s'' created.\n', titleBoxStr, titleDotStr);
    catch ME
        fprintf('Error plotting probability distributions: %s\n', ME.message);
        if ishandle(fh_prob_dist); close(fh_prob_dist); fh_prob_dist = []; end
    end
end