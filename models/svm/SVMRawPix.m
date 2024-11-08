% SVMrawpix.m
clear all
close all

addpath ../../images
addpath SVM-KM

% Load training and test data
[trainingImages, trainingLabels] = loadFaceImages('../../images/face_train.cdataset');
[testingImages, testingLabels] = loadFaceImages('../../images/face_test.cdataset');
numTrainingImages = size(trainingImages, 1);
numTestImages = size(testingImages, 1);
trainingImages = normalize(trainingImages, 'zscore');
testingImages = normalize(testingImages, 'zscore');
fprintf('Loaded augmented training set: %d images\n', size(trainingImages,1));

% Train model
fprintf('Training SVM model on entire image ...\n')
fprintf('Training KNN model on entire image ...\n')
modelSVM = SVMtraining(trainingImages, trainingLabels);

fprintf('Getting model predictions for test set\n');
predictions = zeros(numTestImages, 1);
for i = 1:numTestImages
    testImage = testingImages(i, :);
    predictions(i) = SVMTesting(testImage, modelSVM);
end

fprintf('Evaluating model predictions...\n');
[accuracy, precision, recall, f1_score] = calculateMetrics(predictions, testingLabels);