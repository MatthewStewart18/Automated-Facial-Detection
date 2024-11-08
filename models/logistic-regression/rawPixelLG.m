clear all
close all
addpath ../../images

% Add the folder containing loadFaceImages to the MATLAB path
addpath('/Users/samuelagnew/Documents/year3Term1/csc3067/CSC3067-2425-G3/utils');

% Load training and testing datasets
[trainingImages, trainingLabels] = loadFaceImages("../../images/face_train.cdataset", 1);
[testingImages, testingLabels] = loadFaceImages("../../images/face_test.cdataset", 1);
numTrainingImages = size(trainingImages, 1);
numTestImages = size(testingImages, 1);

% Add the folder containing loadFaceImages to the MATLAB path
addpath('/Users/samuelagnew/Documents/year3Term1/csc3067/CSC3067-2425-G3/feature-extraction-utils/gabor');

%converting -1 to 0
trainingLabels(trainingLabels == -1) = 0;

testingLabels(testingLabels == -1) = 0;

% Fit logistic regression model
mdl = fitglm(trainingImages, trainingLabels, 'Distribution', 'binomial');

% Display the model summary
disp(mdl);

% Predict probabilities for the test data
predictedProbabilities = predict(mdl, testingImages);

% Convert probabilities to binary labels using a threshold of 0.5
predictedLabels = predictedProbabilities >= 0.5;

% Display the predicted labels
disp(predictedLabels);

%converting 0 to -1
predictedLabels(predictedLabels == 0) = -1;
testingLabels(testingLabels == 0) = -1;

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictedLabels, testingLabels);