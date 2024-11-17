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

% pre-process images
train_images = preProcess(train_images, @histEq);
test_images = preProcess(test_images, @histEq);

% Extract features from training images
training_edges = featureExtraction(train_images, @extractEdges);
[training_edges_pca, n_components] = featureExtraction(training_edges, @extractPcaExplainedVar, 0.80);
training_edges_pca = normalize(training_edges_pca, 'zscore');

% Define optimal parameters for the edges & pca SVM model
params = struct('lambda', 1e-20, 'C', Inf, 'kerneloption', 7.6, 'kernel', 'gaussian');
modelSVM = SVMtraining(training_edges_pca, train_labels, params);

% Extract test edges and apply pca with same dim as training
test_edges = featureExtraction(test_images, @extractEdges);
test_edges_pca = featureExtraction(test_edges, @extractPcaDim, n_components);
test_edges_pca = normalize(test_edges_pca, 'zscore');


% Getting model predictions using Gaussian SVM on Edges
predictions = extractPredictionsSVM(test_edges_pca, modelSVM);

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictions, test_labels);