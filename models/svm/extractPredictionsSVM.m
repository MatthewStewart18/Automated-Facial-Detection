function [predictions, confidences] = extractPredictionsSVM(test_features, modelSVM)
    % Initialize output arrays
    predictions = zeros(size(test_features, 1), 1);
    confidences = zeros(size(test_features, 1), 1);
    
    for i = 1:size(test_features, 1)
        testImage = test_features(i, :);
        
        % Get the prediction and distance to decision boundary (confidence)
        [label, confidence] = SVMTesting(testImage, modelSVM); % Modify SVMTesting to return score
        
        predictions(i) = label;
        confidences(i) = confidence;           
    end
end
