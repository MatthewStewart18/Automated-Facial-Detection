clear all;
close all;

% Add required directories to path
addpath ground_truth/annotations
addpath evaluation
addpath ../model
addpath ../../images
addpath ../../utils
addpath ../../detection-images

% Load Model
load('../neural-net/trainedFaceNet.mat', 'net');

% Load image
currentImage = 'im1';
testImage = imread(sprintf('../../detection-images/%s.jpg', currentImage));

% Sliding window parameters
windowSize = [27, 18];
stepSize = [1, 1];    
confidenceThreshold = 0.7;
NMSThreshold = 0.05;

% Run sliding window
[predictions, windowPositions] = windowNN(testImage, windowSize, stepSize, net);

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
visualiseResults(testImage, detections, coords_path, "Neural Net", "");

% show final results
subplot(1,2,2);
ShowDetectionResult(testImage, detections, "Neural Net", "");
