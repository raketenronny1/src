% TEST_PERFORM_INNER_CV_FIELDNAMES
% Simple script verifying that perform_inner_cv accepts both
% fisherFeaturePercent and the deprecated numFisherFeatures field.

addpath('helper_functions');

% Create a tiny synthetic classification problem with 6 probes
% Each probe has two spectra and belongs to class 1 or 3
X = rand(12,10);
y = [ones(6,1); ones(6,1)*3];
probeIDs = repelem((1:6)',2);
wavenumbers = 1:10;

metricNames = {'Accuracy'};
numInnerFolds = 2;

% -- Using fisherFeaturePercent --
pipe = struct();
pipe.feature_selection_method = 'fisher';
pipe.classifier = 'lda';
pipe.hyperparameters_to_tune = {'fisherFeaturePercent'};
pipe.fisherFeaturePercent_range = 0.2;

[hp1, metrics1] = perform_inner_cv(X, y, probeIDs, pipe, wavenumbers, numInnerFolds, metricNames);

% -- Using deprecated numFisherFeatures --
pipe.hyperparameters_to_tune = {'numFisherFeatures'};
pipe.numFisherFeatures_range = 2;

[hp2, metrics2] = perform_inner_cv(X, y, probeIDs, pipe, wavenumbers, numInnerFolds, metricNames);

fprintf('Percent-based field result: %.2f\n', hp1.fisherFeaturePercent);
fprintf('Deprecated field result -> percent: %.2f\n', hp2.fisherFeaturePercent);
