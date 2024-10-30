function prediction = KNNTesting(testImage, modelNN, K)
    distances = EuclideanDistance(modelNN.neighbours, testImage);
    [~, sortedIndices] = sort(distances);
    kNearestIndices = sortedIndices(1:K);
    kNearestLabels = modelNN.labels(kNearestIndices);
    prediction = mode(kNearestLabels);
end