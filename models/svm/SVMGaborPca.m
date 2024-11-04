% gabor_pca_v3.m
clear all;
close all;

% Add required paths
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils/gabor

% Load training and test data
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset');
[test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset');

fprintf('Loaded training set: %d images\n', size(train_images,1));

% Set correct image dimensions
img_height = 27;
img_width = 18;

% Extract Gabor features
fprintf('Extracting Gabor features...\n');
train_features = zeros(size(train_images,1), 19440);

for i = 1:size(train_images,1)
    if mod(i, 10) == 0
        fprintf('Processing training image %d/%d\n', i, size(train_images,1));
    end
    img = reshape(train_images(i,:), [img_height img_width]);
    
    % Extract Gabor features
    features = gabor_feature_vector(img);
    
    % Handle any NaN or Inf values
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
    
    train_features(i, :) = features;
end

% Normalize features before PCA
train_features = normalize(train_features, 'zscore');

% Apply PCA with variance retention
[coeff, score, latent] = pca(train_features);
explained = cumsum(latent)./sum(latent);
n_components = find(explained >= 0.95, 1); % Keep 95% of variance
fprintf('Using %d PCA components\n', n_components);
score = score(:, 1:n_components);

% Train SVM using trainGaborPCASVM
[svm_model, accuracy, precision, recall, f1_score] = trainGaborPCASVM(score, train_labels);

fprintf('Training completed.\n');