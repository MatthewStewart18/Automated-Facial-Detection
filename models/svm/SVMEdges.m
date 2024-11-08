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

% Extract only the first 3 principal components for plotting
training_edges_3D = training_edges(:, 1:3);

% Initialize 3D plot
figure;
hold on;
grid on;

% Define colors for each class (-1 for non-face, 1 for face)
colours = ['r', 'b'];

for i = [-1, 1]  % Non-face is -1, face is 1
    indexes = find(train_labels == i);
    scatter3(training_edges_3D(indexes, 1), training_edges_3D(indexes, 2), training_edges_3D(indexes, 3), ...
        30, colours((i + 3) / 2), 'filled'); % Adjust marker size and color here
end

% Label axes and add legend
xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');
legend('Non-face', 'Face');
title('3D Plot of First 3 Principal Components for Face Classification');
hold off;


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