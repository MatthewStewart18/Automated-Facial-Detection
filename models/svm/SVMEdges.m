clear all;
close all;

% Add required paths
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils/edges/
addpath ../../preprocessing-utils/hist-eq/
addpath ../../preprocessing-utils/
addpath SVM-KM/

% Load training and test data
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset');
[test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset');
fprintf('Loaded training set: %d images\n', size(train_images,1));

% preProcess the images
% [train_images, test_images] = preProcessImages(train_images, test_images);

% define mask
training_edges = extractEdges(train_images);

% Train model on reduced dimension of edges
modelSVM = SVMtraining(training_edges, train_labels);

% Extract test edges
test_edges = extractEdges(test_images);

% Getting model predictions using Gaussian SVM on Edges
predictions = extractPredictionsSVM(test_edges, modelSVM);

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictions, test_labels);

% Display images of the correct/incorrect predictions
dispPreds(predictions, test_labels, test_images);

% Save the trained KNN model to a .mat file
save('modelSVM.mat', 'modelSVM');