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
[training_edges, n_components] = pcaExplainedVar(training_edges, 0.80);

% Train model on reduced dimension of edges
modelSVM = SVMtraining(training_edges, train_labels);

% Extract test edges
test_edges = extractEdges(test_images);

% Apply pca with same dim as training
test_edges = pcaByDimension(test_edges, n_components);

% Getting model predictions using Gaussian SVM on Edges
predictions = extractPredictionsSVM(test_edges, modelSVM);

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictions, test_labels);


% B1 = [-1, -1, -1; 0, 0, 0; 1, 1, 1];
% B2 = B1';
% [edges, ~, ~] = edgeExtraction(img,B1,B2);
% subplot(1,2,1), imagesc(uint8(img)), axis image, colormap("gray");
% subplot(1,2,2), imagesc(edges), axis image, colormap("gray");