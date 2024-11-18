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

% Choose the feature extraction type
featureExtractorFunc = @extractEdges;

% reduce noise in images
% train_images = preProcess(train_images, @medianFilter, 2);
% test_images = preProcess(test_images, @medianFilter, 2);
% apply histogram equalisation
train_images = preProcess(train_images, @histEq);
test_images = preProcess(test_images, @histEq);

% Extract features from training images
training_edges = featureExtraction(train_images, featureExtractorFunc);

% Define optimal parameters for the edge-based SVM model
params = struct('kerneloption', 3.75, 'kernel', 'polyhomog');
modelSVM = SVMtraining(training_edges, train_labels, params);

% Extract test edges
test_edges = featureExtraction(test_images, featureExtractorFunc);
predictions = extractPredictionsSVM(test_edges, modelSVM);

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictions, test_labels);

% Display images of the correct/incorrect predictions
dispPreds(predictions, test_labels, test_images);

% Save the trained SVM model to a .mat file
save('saved-models/modelSVMEdges.mat', 'modelSVM');