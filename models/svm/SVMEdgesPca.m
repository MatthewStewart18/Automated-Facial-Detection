clear all;
close all;

% Add required paths
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils/edges/
addpath ../../preprocessing-utils/hist-eq/

% Load training and test data
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset');
[test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset');
fprintf('Loaded training set: %d images\n', size(train_images,1));

% define mask
training_edges = extractEdges(train_images);

% Apply PCA with 80% variance retention
[training_edges, n_components] = extractPcaExplainedVar(training_edges, 0.80);

% Train model on reduced dimension of edges
modelSVM = SVMtraining(training_edges, train_labels);

% Extract test edges
test_edges = extractEdges(test_images);

% Apply pca with same dim as training
test_edges = extractPcaDim(test_edges, n_components);

% Getting model predictions using Gaussian SVM on Edges
predictions = extractPredictionsSVM(test_edges, modelSVM);

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictions, test_labels);