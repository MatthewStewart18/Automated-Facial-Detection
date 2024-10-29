function prediction = NNTesting(testImage, modelNN)
    distances = EuclideanDistance(modelNN.neighbours, testImage);
    [~, minIndex] = min(distances);
    prediction = modelNN.labels(minIndex);
end
