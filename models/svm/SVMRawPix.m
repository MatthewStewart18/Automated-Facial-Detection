% SVMrawpix.m
clear all
close all
addpath SVM-KM
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils
addpath ../../preprocessing-utils

% Load training and test data
[trainingImages, trainingLabels] = loadFaceImages('../../images/face_train.cdataset');
[testingImages, testingLabels] = loadFaceImages('../../images/face_test.cdataset');

% Train model
params = struct('lambda', 1e-20, 'C', Inf, 'kerneloption', 6, 'kernel', 'poly');
modelSVM = SVMtraining(trainingImages, trainingLabels, params);

predictions = extractPredictionsSVM(testingImages, modelSVM);

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictions, testingLabels);

% Display images of the correct/incorrect predictions
dispPreds(predictions, testingLabels, testingImages);
% save('modelSVM.mat', 'modelSVM');