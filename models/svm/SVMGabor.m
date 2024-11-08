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
numTrainingImages = size(train_images, 1);
numTestImages = size(test_images, 1);

% Set correct image dimensions
imgHeight = 27;
imgWidth = 18;


% Extract optimized Gabor features
fprintf('Extracting optimized Gabor features...\n');
trainingFeatureSet = zeros(size(train_images,1), 2000); % Adjusted for 2000 selected features
feature_mask = []; % Will store the feature mask for consistent feature selection

for i = 1:numTrainingImages
    img = reshape(train_images(i,:), [imgHeight imgWidth]);
    
    features = gabor_feature_vector_optimised(img);

    % Handle any NaN or Inf values
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
    
    trainingFeatureSet(i, :) = features;

end

trainingFeatureSet = normalize(trainingFeatureSet, 'zscore');

% Train the model
modelSVM = SVMtraining(trainingFeatureSet, train_labels);

% Extract Gabor features for test set
testingFeatureSet = zeros(size(test_images, 1), 2000);

for i = 1:numTestImages
    img = reshape(test_images(i,:), [imgHeight imgWidth]);
    
    features = gabor_feature_vector_optimised(img);

    % Handle any NaN or Inf values
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
    
    testingFeatureSet(i, :) = features;
end

testingFeatureSet = normalize(testingFeatureSet, 'zscore');

fprintf('Getting model predictions');
predictions = zeros(numTestImages);
for i = 1:numTestImages
    testImage = testingFeatureSet(i, :);
    predictions(i) = SVMTesting(testImage, modelSVM);
end

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictions, test_labels);