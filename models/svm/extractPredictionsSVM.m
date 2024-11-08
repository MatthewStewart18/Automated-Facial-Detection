function [predictions] = extractPredictionsSVM(test_features, modelSVM)
predictions = zeros(size(test_features, 1));
for i = 1:size(test_features, 1)
    testImage = test_features(i, :);
    predictions(i, 1) = SVMTesting(testImage, modelSVM);
end
end

