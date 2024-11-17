clear all;
close all;

% Add required paths
addpath SVM-KM
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils
addpath ../../preprocessing-utils

% Load training and test data
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset');
[test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset');
fprintf('Loaded training set: %d images\n', size(train_images,1));

% Choose the feature extraction and pre-processing methods
featureExtractorFunc = @extractEdges;
preprocessingFunc = @histEq;

% pre-process images
train_images = preProcess(train_images, preprocessingFunc);
test_images = preProcess(test_images, preprocessingFunc);

% Extract features from training images
training_edges = featureExtraction(train_images, featureExtractorFunc);

% Train model on reduced dimension of edges
modelSVM = SVMtraining(training_edges, train_labels);

% Extract test edges
test_edges = featureExtraction(test_images, featureExtractorFunc);

% Getting model predictions using Gaussian SVM on Edges
predictions = extractPredictionsSVM(test_edges, modelSVM);

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictions, test_labels);

% Display images of the correct/incorrect predictions
dispPreds(predictions, test_labels, test_images);

% Save the trained KNN model to a .mat file
save('modelSVM.mat', 'modelSVM');