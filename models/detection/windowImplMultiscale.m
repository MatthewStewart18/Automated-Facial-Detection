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
preprocessingType = PreprocessingType.HistEq;

% Load Model
path = sprintf('../model/saved-models/%s/%s_%s_ModelTest.mat', ...
    char(modelType), char(featureType), char(preprocessingType));
load(path, 'model');

% Load image
currentImage = 'im1';
testImage = imread(sprintf('../../detection-images/%s.jpg', currentImage));

% Parameters
windowSize = [27, 18];
stepSize = [1, 1];    
confidenceThreshold = 0.67;
NMSThreshold = 0.05;
scales = [0.9, 1, 1.1];

finalDetections = [];
allScaleStats = struct('scale', {}, 'numDetections', {}, 'highConfDetections', {});

% Process each scale separately
tic
for scaleIdx = 1:length(scales)
    scale = scales(scaleIdx);
    fprintf('Processing scale: %.2f\n', scale);
    
    % Scale the image
    scaledImage = imresize(testImage, scale, 'bicubic');
    
    % Run detection
    [predictions, windowPositions] = window(scaledImage, windowSize, stepSize, model);
    
    % Convert positions back to original coordinates
    windowPositions = windowPositions / scale;
    
    % Scale confidence of predictions to [0, 1]
    predictions(:, 2) = rescale(predictions(:, 2));
    
    % Create bboxes for this scale
    scaledWidth = windowSize(2) / scale;
    scaledHeight = windowSize(1) / scale;
    
    % Filter by confidence
    highConfidenceIndices = predictions(:, 2) >= confidenceThreshold;
    filteredPreds = predictions(highConfidenceIndices, :);
    filteredPos = windowPositions(highConfidenceIndices, :);
    
    % Create bbox for this scale
    bbox = [filteredPos(:, 1:2), repmat([scaledWidth, scaledHeight], size(filteredPos, 1), 1), ...
            filteredPreds(:, 2)];
            
    % Apply NMS to this scale's detections
    scaleDetections = simpleNMS(bbox, NMSThreshold);
    
    % Store statistics for this scale
    allScaleStats(scaleIdx).scale = scale;
    allScaleStats(scaleIdx).numDetections = size(bbox, 1);
    allScaleStats(scaleIdx).highConfDetections = sum(bbox(:, 5) >= 0.9);
    
    % Accumulate detections
    finalDetections = [finalDetections; scaleDetections];
end
detectionTime = toc;

% Final NMS pass on accumulated detections with slightly more lenient threshold
if size(finalDetections, 1) > 0
    finalDetections = simpleNMS(finalDetections, NMSThreshold * 3);
end

% Print detection statistics
fprintf('\nDetection Results:\n');
fprintf('Total detection time: %.2f seconds\n', detectionTime);
fprintf('Final number of detections after combined NMS: %d\n', size(finalDetections, 1));
fprintf('High confidence final detections (>= 0.9): %d\n\n', sum(finalDetections(:, 5) >= 0.9));

% Print per-scale statistics
fprintf('Per-scale statistics:\n');
for i = 1:length(allScaleStats)
    fprintf('Scale %.2f:\n', allScaleStats(i).scale);
    fprintf('  - Detections before NMS: %d\n', allScaleStats(i).numDetections);
    fprintf('  - High confidence detections: %d\n', allScaleStats(i).highConfDetections);
end

% Visualize results
figure;
subplot(1,2,1);
coords_path = sprintf('ground_truth/annotations/%s.txt', currentImage);
visualiseResults(testImage, finalDetections, coords_path, modelType, featureType);

subplot(1,2,2);
ShowDetectionResult(testImage, finalDetections, modelType, featureType);