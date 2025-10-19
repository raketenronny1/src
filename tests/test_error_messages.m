function tests = test_error_messages
%TEST_ERROR_MESSAGES Ensure critical error messages provide actionable guidance.
%
%   Run with: results = runtests('tests/test_error_messages.m');
%
%   These checks focus on key helper functions whose errors commonly reach
%   end users when configuration drifts occur. Each test asserts that the
%   thrown message contains our troubleshooting guidance text.

    tests = functiontests(localfunctions);
end

function testApplyModelDimensionMismatch(testCase)
    import matlab.unittest.constraints.ContainsSubstring

    model = struct();
    model.featureSelectionMethod = 'pca';
    model.PCAMu = ones(1,5);
    model.PCACoeff = eye(5);
    model.LDAModel = []; % Not used because error is thrown earlier.

    X = rand(2,3);
    wn = 1:3;

    try
        apply_model_to_data(model, X, wn);
        testCase.verifyFail('Expected dimension mismatch error was not thrown.');
    catch ME
        testCase.verifyEqual(ME.identifier, 'apply_model_to_data:DimensionMismatch');
        testCase.verifyThat(ME.message, ContainsSubstring('Troubleshooting tip: ensure new spectra are preprocessed'));
    end
end

function testApplyModelMissingFeatureIndices(testCase)
    import matlab.unittest.constraints.ContainsSubstring

    model = struct();
    model.featureSelectionMethod = 'fisher';
    model.LDAModel = [];

    X = rand(4,4);
    wn = 1:4;

    try
        apply_model_to_data(model, X, wn);
        testCase.verifyFail('Expected missing feature indices error was not thrown.');
    catch ME
        testCase.verifyEqual(ME.identifier, 'apply_model_to_data:MissingFeatureIndices');
        testCase.verifyThat(ME.message, ContainsSubstring('Troubleshooting tip: confirm the training pipeline saved this field'));
    end
end

function testConfigureCfgNameValuePairs(testCase)
    import matlab.unittest.constraints.ContainsSubstring

    try
        configure_cfg('projectRoot');
        testCase.verifyFail('Expected name-value pair error was not thrown.');
    catch ME
        testCase.verifyEqual(ME.identifier, 'configure_cfg:NameValuePairs');
        testCase.verifyThat(ME.message, ContainsSubstring('Troubleshooting tip: check configure_cfg calls for a missing value'));
    end
end

function testComputePCATooFewRows(testCase)
    import matlab.unittest.constraints.ContainsSubstring

    try
        compute_pca_t2_q([], 0.05, 0.95);
        testCase.verifyFail('Expected insufficient data error was not thrown.');
    catch ME
        testCase.verifyEqual(ME.identifier, 'compute_pca_t2_q:InsufficientData');
        testCase.verifyThat(ME.message, ContainsSubstring('Troubleshooting tip: supply additional spectra'));
    end
end
