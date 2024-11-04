% SVMGabor.m
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

% Extract optimized Gabor features
fprintf('Extracting optimized Gabor features...\n');
train_features = zeros(size(train_images,1), 2000); % Adjusted for 2000 selected features
feature_mask = []; % Will store the feature mask for consistent feature selection

for i = 1:size(train_images,1)
    if mod(i, 10) == 0
        fprintf('Processing training image %d/%d\n', i, size(train_images,1));
    end
    img = reshape(train_images(i,:), [img_height img_width]);
    
    try
        % Extract and normalize optimized Gabor features
        if isempty(feature_mask)
            [features, feature_mask] = gabor_feature_vector_optimised(img);
        else
            [features, ~] = gabor_feature_vector_optimised(img, feature_mask);
        end
        
        % Handle any NaN or Inf values
        features(isnan(features)) = 0;
        features(isinf(features)) = 0;
        
        train_features(i, :) = features;
    catch ME
        fprintf('Error processing image %d: %s\n', i, ME.message);
        rethrow(ME);
    end
end

% Train SVM using modified trainGaborSVM
[svm_model, accuracy, precision, recall, f1_score] = trainGaborSVM(train_features, train_labels);

fprintf('Training completed.\n');

% Store feature mask in the model for future use
svm_model.feature_mask = feature_mask;

% Save the model and feature mask
save('svm_model.mat', 'svm_model');