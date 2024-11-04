% SVMrawpix.m
clear all
close all
addpath ../../images
addpath SVM-KM

% Load training and test data
[trainingImages, trainingLabels] = loadFaceImages('../../images/face_train.cdataset');
[testingImages, testingLabels] = loadFaceImages('../../images/face_test.cdataset');
numTrainingImages = size(trainingImages, 1);
numTestImages = size(testingImages, 1);
fprintf('Loaded augmented training set: %d images\n', size(trainingImages,1));

% Convert labels to -1/1 for SVM (only necessary conversion for SVM)
trainingLabels(trainingLabels == 0) = -1;
testingLabels(testingLabels == 0) = -1;

% Train model
fprintf('Training SVM model on entire image ...\n')
% Basic SVM parameters
C = 15;
lambda = 1e-5;
kernel = 'gaussian';
sigma = 1.5;

[xsup, w, w0] = svmclass(trainingImages, trainingLabels, C, lambda, kernel, sigma, 1);

% Create model structure
model = struct();
model.type = 'binary';
model.xsup = xsup;
model.w = w;
model.w0 = w0;
model.param.kernel = kernel;
model.param.sigmakernel = sigma;

fprintf('Getting model predictions for test set\n');
predictions = zeros(numTestImages, 1);
for i = 1:numTestImages
    testImage = testingImages(i, :);
    [predictions(i), ~] = SVMTesting(testImage, model);
end

fprintf('Evaluating model predictions...\n');
[accuracy, precision, recall, f1] = calculateMetrics(predictions, testingLabels);

% Print results in same format as KNN
fprintf('\nDetailed Confusion Matrix:\n');
TP = sum(predictions == 1 & testingLabels == 1);
TN = sum(predictions == -1 & testingLabels == -1);
FP = sum(predictions == 1 & testingLabels == -1);
FN = sum(predictions == -1 & testingLabels == 1);

fprintf('True Positives (TP): %d\n', TP);
fprintf('True Negatives (TN): %d\n', TN);
fprintf('False Positives (FP): %d\n', FP);
fprintf('False Negatives (FN): %d\n', FN);

fprintf('\nClassification Results:\n');
fprintf('Accuracy: %.2f%%\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1);