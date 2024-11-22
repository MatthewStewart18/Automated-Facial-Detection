clear all;
close all;

% Add required directories to path
addpath ../svm
addpath ../svm/SVM-KM
addpath ../knn
addpath ../model
addpath ../../images
addpath ../../utils
addpath ../../detection-images
addpath ../../preprocessing-utils
addpath ../../feature-extraction-utils
addpath ../../feature-extraction-utils/feature-extractors

% Set current Model
modelType = ModelType.SVM;
featureType = FeatureType.Edges;
preprocessingType = PreprocessingType.HistEq;

% Load Model
path = sprintf('../model/saved-models/%s/%s_%s_Model.mat', ...
    char(modelType), char(featureType), char(preprocessingType));
load(path, 'model');

% Load image
testImage = imread('../../detection-images/im4.jpg');

% Sliding window parameters
windowSize = [27, 18];
stepSize = [1, 1];    
confidenceThreshold = 0.9;
NMSThreshold = 0.5;

% Run sliding window
[predictions, windowPositions] = window(testImage, windowSize, stepSize, model);

% Normalize confidence to [0, 1]
predictions(:, 2) = max(predictions(:, 2), 0);
minConfidence = min(predictions(:, 2)); % Get minimum confidence value
maxConfidence = max(predictions(:, 2)); % Get maximum confidence value
scaledConfidence = (predictions(:, 2) - minConfidence) / (maxConfidence - minConfidence);

% Filter predictions based on confidence threshold
highConfidenceIndices = scaledConfidence >= confidenceThreshold;
filteredPredictions = predictions(highConfidenceIndices, :);
filteredPositions = windowPositions(highConfidenceIndices, :);

% reformat positions to bbox
bbox = [filteredPositions(:, 1:2), repmat([windowSize(2), windowSize(1)], size(filteredPositions, 1), 1), ...
         filteredPredictions(:, 2)];

% Display results
fprintf('Total windows evaluated: %d\n', size(predictions, 1));
fprintf('Windows with high confidence (>= %.2f): %d\n', confidenceThreshold, sum(highConfidenceIndices));

% Apply Non-Maximum Suppression
detections = simpleNMS(bbox, NMSThreshold);
% Display results
ShowDetectionResult(testImage, detections);
