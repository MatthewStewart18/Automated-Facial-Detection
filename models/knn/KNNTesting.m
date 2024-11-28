function [predictions, confidences] = KNNTesting(testImages, modelKNN)
    % Initialize arrays for predictions and confidence scores
    numImages = size(testImages, 1); % Assume each row is an image
    predictions = zeros(numImages, 1); % Preallocate predictions array
    confidences = zeros(numImages, 1); % Preallocate confidence array
    K = extractfield(modelKNN.K, 'K');
    
    % Iterate through each test image
    for i = 1:numImages
        testImage = testImages(i, :); % Extract the i-th image
        distances = EuclideanDistance(modelKNN.neighbours, testImage); % Compute distances
        [~, sortedIndices] = sort(distances); % Sort by closest distance
        kNearestIndices = sortedIndices(1:K); % Get indices of K nearest neighbors
        kNearestLabels = modelKNN.labels(kNearestIndices); % Labels of the K nearest neighbors
        
        % Predict the class based on the majority vote
        predictedLabel = mode(kNearestLabels);
        predictions(i) = predictedLabel;
        
        % Calculate confidence as the proportion of neighbors with the predicted label
        confidence = sum(kNearestLabels == predictedLabel) / K;
        confidences(i) = confidence; % Store the confidence score
    end
end
