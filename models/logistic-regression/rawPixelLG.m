clear all
close all
addpath ../../images
addpath ../../utils

% Load training and testing datasets
[trainingImages, trainingLabels] = loadFaceImages("../../images/face_train.cdataset", 1);
[testingImages, testingLabels] = loadFaceImages("../../images/face_test.cdataset", 1);
numTrainingImages = size(trainingImages, 1);
numTestImages = size(testingImages, 1);

%converting -1 to 0
trainingLabels(trainingLabels == -1) = 0;
testingLabels(testingLabels == -1) = 0;

% Fit logistic regression model
mdl = fitglm(trainingImages, trainingLabels,'Distribution', 'binomial');

% Predict probabilities for the test data
predictedProbabilities = predict(mdl, testingImages);

% Convert probabilities to binary labels using a threshold of 0.5
predictedLabels = double(predictedProbabilities >= 0.5);

% Display the predicted labels
disp(predictedLabels);

%converting 0 to -1
predictedLabels(predictedLabels == 0) = -1;
testingLabels(testingLabels == 0) = -1;

fprintf('Evaluating model predictions...\n');
[accuracy, precision, recall, f1_score, confusionMatrix] = calculateMetrics(predictedLabels, testingLabels);


createRatioBarChartSVM(confusionMatrix, "Raw Pixel Logistic-Regression", accuracy, precision, recall,f1_score)
