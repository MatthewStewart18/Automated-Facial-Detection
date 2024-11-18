clear all;
close all;

% Add required paths
addpath SVM-KM
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils/feature-extractors
addpath ../../preprocessing-utils

% Load training and test data
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset');
[test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset');
fprintf('Loaded training set: %d images\n', size(train_images,1));

% Choose the feature extraction and pre-processing methods
featureExtractorFunc = @extractHog;
preprocessingFunc = @histEq;

% pre-process images
train_images = preProcess(train_images, preprocessingFunc);
test_images = preProcess(test_images, preprocessingFunc);

% Extract features from training images
training_hog = featureExtraction(train_images, featureExtractorFunc);

% Define optimal parameters for the edge-based SVM model
params = struct('kerneloption', 8, 'kernel', 'polyhomog');
modelSVM = SVMtraining(training_hog, train_labels, params);

% Extract test edges
test_hog = featureExtraction(test_images, featureExtractorFunc);
predictions = extractPredictionsSVM(test_hog, modelSVM);

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictions, test_labels);

% Display images of the correct/incorrect predictions
% dispPreds(predictions, test_labels, test_images);

% Save the trained SVM model to a .mat file
save('modelSVM.mat', 'modelSVM');