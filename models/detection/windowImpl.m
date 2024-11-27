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
modelType = ModelType.LG;
featureType = FeatureType.HOG;
preprocessingType = PreprocessingType.HistEq;

% Load Model
path = sprintf('../model/saved-models/%s/%s_%s_Model.mat', ...
    char(modelType), char(featureType), char(preprocessingType));
load(path, 'model');

% Load image
currentImage = 'im1';
testImage = imread(sprintf('../../detection-images/%s.jpg', currentImage));

% Sliding window parameters
windowSize = [27, 18];
stepSize = [1, 1];    
confidenceThreshold = 0.7;
NMSThreshold = 0.05;

% Run sliding window
[predictions, windowPositions] = window(testImage, windowSize, stepSize, model);

% Scale confidence of model predictions to [0, 1]
predictions(:, 2) = rescale(predictions(:, 2));

% Filter predictions based on confidence threshold
highConfidenceIndices = predictions(:, 2) >= confidenceThreshold;
filteredPredictions = predictions(highConfidenceIndices, :);
filteredPositions = windowPositions(highConfidenceIndices, :);

% reformat positions to bbox
bbox = [filteredPositions(:, 1:2), repmat([windowSize(2), windowSize(1)], size(filteredPositions, 1), 1), ...
         filteredPredictions(:, 2)];

% Display results
fprintf('Total windows evaluated: %d\n', size(predictions, 1));
fprintf('Windows above confidence threshold (>= %.2f): %d\n', confidenceThreshold, sum(highConfidenceIndices));

% Apply Non-Maximum Suppression
detections = simpleNMS(bbox, NMSThreshold);

% show final results
ShowDetectionResult(testImage, detections);
