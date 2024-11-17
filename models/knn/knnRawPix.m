clear all
close all
addpath ../../images
addpath ../../utils

% Load training and test data
[trainingImages, trainingLabels] = loadFaceImages('../../images/face_train.cdataset');
[testingImages, testingLabels] = loadFaceImages('../../images/face_test.cdataset');
numTrainingImages = size(trainingImages, 1);
numTestImages = size(testingImages, 1);
fprintf('Loaded augmented training set: %d images\n', size(trainingImages,1));

% Train model
fprintf('Training KNN model on entire image ...\n')
modelKNN = NNtraining(trainingImages, trainingLabels);

% Set K to sqrt(N)
K = round(sqrt(numTrainingImages)); 
% this isnt working great, using 50 temporarily
% K = 50;

fprintf('Getting model predictions for K = %d\n', K);
predictions = zeros(numTestImages);
for i = 1:numTestImages
    testImage = testingImages(i, :);
    predictions(i, 1) = KNNTesting(testImage, modelKNN, K);
end

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictions, testingLabels);

% Display images of the correct/incorrect predictions
dispPreds(predictions(:, 1), testingLabels, testingImages);

% Save the trained KNN model to a .mat file
save('modelKNN.mat', 'modelKNN', 'K');