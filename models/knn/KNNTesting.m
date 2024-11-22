function predictions = KNNTesting(testImages, modelKNN)
    % Initialize the array to store predictions
    numImages = size(testImages, 1); % Assume each row is an image
    predictions = zeros(numImages, 1); % Preallocate predictions array
    % K = extractfield(modelKNN.K, 'K');
    K = 5;
    
    % Iterate through each test image
    for i = 1:numImages
        testImage = testImages(i, :); % Extract the i-th image
        distances = EuclideanDistance(modelKNN.neighbours, testImage);
        [~, sortedIndices] = sort(distances);
        kNearestIndices = sortedIndices(1:K);
        kNearestLabels = modelKNN.labels(kNearestIndices);
        predictions(i) = mode(kNearestLabels); % Store prediction
    end
end
