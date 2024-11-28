clear all;
close all;

% Add required directories to path
addpath ground_truth/annotations
addpath evaluation
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
featureType = FeatureType.EdgesPCA;
preprocessingType = PreprocessingType.None;

% Load Model
path = sprintf('../model/saved-models/%s/%s_%s_Model.mat', ...
    char(modelType), char(featureType), char(preprocessingType));
load(path, 'model');

% Load image
currentImage = 'im1';
testImage = imread(sprintf('../../detection-images/%s.jpg', currentImage));
image_path = '../../detection-images/im1.jpg';

% Sliding window parameters
windowSize = [27, 18];
stepSize = [1, 1];    
confidenceThreshold = 0.7;
NMSThreshold = 0.05;

% if ~exist('ground-truth/annotations/im1.txt', 'file')
%     createGroundTruthAnnotations(image_path, 'ground_truth/annotations/im1.txt');
% end

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

figure;
subplot(1,2,1);
coords_path = sprintf('ground_truth/annotations/%s.txt', currentImage);
visualizeResults(testImage, detections, coords_path);
% show final results
subplot(1,2,2);
ShowDetectionResult(testImage, detections);