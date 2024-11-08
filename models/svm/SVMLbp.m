% lbp_pca_v3.m
clear all;
close all;

% Add required paths
addpath ../../images
addpath ../../utils

% Load training and test data
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset');
[test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset');
numTrainingImages = size(train_images, 1);
numTestImages = size(test_images, 1);
fprintf('Loaded training set: %d images\n', size(train_images,1));

% Set correct image dimensions
img_height = 27;
img_width = 18;

% Extract LBP features
fprintf('Extracting LBP features...\n');
cellSize = [6 6];  % Adjust cell size for more features
numNeighbors = 10; 
radius = 1;       

% Get size of LBP features for one image to initialize matrix
temp_img = reshape(train_images(1,:), [img_height img_width]);
temp_features = extractLBPFeatures(temp_img, 'CellSize', cellSize, ...
    'NumNeighbors', numNeighbors, 'Radius', radius);
feature_length = length(temp_features);

% Initialize feature matrices
train_features = zeros(size(train_images,1), feature_length);
train_images = normalize(train_images, 'zscore');

for i = 1:size(train_images,1)
    if mod(i, 10) == 0
        fprintf('Processing training image %d/%d\n', i, size(train_images,1));
    end
    img = reshape(train_images(i,:), [img_height img_width]);
    
    % Extract LBP features
    features = extractLBPFeatures(img, 'CellSize', cellSize, ...
        'NumNeighbors', numNeighbors, 'Radius', radius);
    
    train_features(i, :) = features;
end

% Normalize features before PCA
train_features = normalize(train_features, 'zscore');

% Apply PCA with variance retention
[coeff, score, latent] = pca(train_features);
explained = cumsum(latent)./sum(latent);
n_components = find(explained >= 0.95, 1); % Keep 95% of variance
fprintf('Using %d PCA components\n', n_components);

train_features = score(:, 1:n_components);
train_features = normalize(train_features, 'zscore');

% Train the model 
modelSVM = SVMtraining(train_features, train_labels);

% Extract LBP features from test images
test_features = zeros(size(test_images,1), feature_length);
test_images = normalize(test_images, 'zscore');

for i = 1:size(test_images,1)
    if mod(i, 10) == 0
        fprintf('Processing testing image %d/%d\n', i, size(test_images,1));
    end
    img = reshape(test_images(i,:), [img_height img_width]);
    
    % Extract LBP features
    features = extractLBPFeatures(img, 'CellSize', cellSize, ...
        'NumNeighbors', numNeighbors, 'Radius', radius);
    
    test_features(i, :) = features;
end

% Apply PCA transformation to test features using training PCA parameters
test_features = normalize(test_features, 'zscore');
test_features = test_features * coeff(:,1:n_components);
test_features = normalize(test_features, 'zscore');

% Get the predictions
fprintf('Getting model predictions for test set\n');
predictions = zeros(numTestImages, 1);

for i = 1:numTestImages
    testImage = test_features(i, :);
    predictions(i) = SVMTesting(testImage, modelSVM);
end

fprintf('Evaluating model predictions...\n');
[accuracy, precision, recall, f1_score] = calculateMetrics(predictions, test_labels);
fprintf('Training completed.\n');