classdef test_helper_functions < matlab.unittest.TestCase
    % Unit tests for helper functions related to spectral processing.

    methods (TestClassSetup)
        function addHelperPaths(~)
            testsFolder = fileparts(mfilename('fullpath'));
            projectRoot = fileparts(testsFolder);
            addpath(fullfile(projectRoot, 'helper_functions'));
        end
    end

    methods (Test)
        function testBinSpectraAveragesCorrectly(testCase)
            spectra = [1 3 5 7; 2 4 6 8];
            wavenumbers = [100 101 102 103];
            [spectra_binned, wavenumbers_binned] = bin_spectra(spectra, wavenumbers, 2);

            expectedSpectra = [2 6; 3 7];
            expectedWavenumbers = [100.5 102.5];

            testCase.verifyEqual(spectra_binned, expectedSpectra, 'AbsTol', 1e-12);
            testCase.verifyEqual(wavenumbers_binned, expectedWavenumbers, 'AbsTol', 1e-12);
        end

        function testBinSpectraNoChangeWhenFactorLeqOne(testCase)
            spectra = [1 2 3; 4 5 6];
            wavenumbers = [10 11 12];

            [binnedSpectra, binnedWavenumbers] = bin_spectra(spectra, wavenumbers, 1);
            testCase.verifyEqual(binnedSpectra, spectra);
            testCase.verifyEqual(binnedWavenumbers, wavenumbers);

            [binnedSpectraZero, binnedWavenumbersZero] = bin_spectra(spectra, wavenumbers, 0);
            testCase.verifyEqual(binnedSpectraZero, spectra);
            testCase.verifyEqual(binnedWavenumbersZero, wavenumbers);
        end

        function testBinSpectraHandlesColumnWavenumbers(testCase)
            spectra = [1 2 3 4];
            wavenumbers = (100:103)';

            [binnedSpectra, binnedWavenumbers] = bin_spectra(spectra, wavenumbers, 2);

            testCase.verifySize(binnedSpectra, [1 2]);
            testCase.verifyTrue(isrow(binnedWavenumbers));
            testCase.verifyEqual(binnedWavenumbers, [100.5 102.5], 'AbsTol', 1e-12);
        end

        function testBinSpectraErrorsOnMismatchedLengths(testCase)
            spectra = [1 2 3 4; 5 6 7 8];
            wavenumbers = [100 101 102];

            didError = false;
            try
                bin_spectra(spectra, wavenumbers, 2);
            catch ME
                didError = true;
                testCase.verifyTrue(contains(ME.message, 'Number of spectral features must match'), ...
                    'Unexpected error message for mismatched wavenumbers length.');
            end
            testCase.verifyTrue(didError, 'bin_spectra should error when wavenumber length mismatches spectral features.');
        end

        function testBinSpectraHandlesLargeBinningFactor(testCase)
            spectra = [1 2 3];
            wavenumbers = [100 101 102];

            [binnedSpectra, binnedWavenumbers] = bin_spectra(spectra, wavenumbers, 5);

            testCase.verifySize(binnedSpectra, [1 0]);
            testCase.verifySize(binnedWavenumbers, [1 0]);
        end

        function testCalculateFisherRatioBasic(testCase)
            X = [1 2 3; 4 5 6; 7 8 9; 10 11 12];
            y = [0; 0; 1; 1];

            fisher = calculate_fisher_ratio(X, y);

            class1Mean = mean(X(1:2, :), 1);
            class2Mean = mean(X(3:4, :), 1);
            class1Var = var(X(1:2, :), 0, 1);
            class2Var = var(X(3:4, :), 0, 1);
            expected = (class1Mean - class2Mean).^2 ./ (class1Var + class2Var);

            testCase.verifyEqual(fisher, expected, 'AbsTol', 1e-12);
        end

        function testCalculateFisherRatioZeroVarianceGivesInf(testCase)
            X = [1 1 1; 1 1 1; 3 3 3; 3 3 3];
            y = [0; 0; 1; 1];

            fisher = calculate_fisher_ratio(X, y);

            testCase.verifyTrue(all(isinf(fisher)), 'Expected infinite Fisher ratio when classes are perfectly separated with zero variance.');
        end

        function testCalculateFisherRatioSingleSampleClass(testCase)
            X = [1 2 3; 4 5 6; 7 8 9];
            y = [0; 1; 1];

            fisher = calculate_fisher_ratio(X, y);

            class1Mean = mean(X(1, :), 1);
            class2Mean = mean(X(2:3, :), 1);
            class2Var = var(X(2:3, :), 0, 1);
            expected = (class1Mean - class2Mean).^2 ./ (class2Var);

            testCase.verifyEqual(fisher, expected, 'AbsTol', 1e-12);
        end

        function testCalculateFisherRatioErrorsWithMoreThanTwoClasses(testCase)
            X = [1 2; 3 4; 5 6];
            y = [0; 1; 2];

            didError = false;
            try
                calculate_fisher_ratio(X, y);
            catch ME
                didError = true;
                testCase.verifyTrue(contains(ME.message, 'requires exactly two classes'), ...
                    'Unexpected error message when more than two classes are provided.');
            end
            testCase.verifyTrue(didError, 'calculate_fisher_ratio should error when more than two classes are provided.');
        end
    end
end
