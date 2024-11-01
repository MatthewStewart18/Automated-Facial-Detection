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

featureMatrixTraining = zeros(numTrainingImages, 3888);
featureMatrixTest = zeros(numTestImages, 3888);


% Add the folder containing loadFaceImages to the MATLAB path
addpath('/Users/samuelagnew/Documents/year3Term1/csc3067/CSC3067-2425-G3/feature-extraction-utils/gabor');

imageListTraining = reshape2dImage(trainingImages);
for imageNumTraining = 1:length(imageListTraining)
    image = cell2mat(imageListTraining(imageNumTraining));
    featureMatrixTraining(imageNumTraining, :) = gabor_feature_vector_subset(image);
end
[~, ~, featureMatrixTraining] = pca(featureMatrixTraining, 20);

imageListTest = reshape2dImage(testingImages);
for imageNumTest = 1:length(imageListTest)
    image = cell2mat(imageListTest(imageNumTest));
    featureMatrixTest(imageNumTest, :) = gabor_feature_vector_subset(image);
end
[~, ~, featureMatrixTest] = pca(featureMatrixTest, 20);



featureMatrixTrainingLabeld = [featureMatrixTraining, trainingLabels];

featureMatrixTestLabeld = [featureMatrixTest, testingLabels];

%converting -1 to 0
trainingLabels(trainingLabels == -1) = 0;

testingLabels(testingLabels == -1) = 0;

% Fit logistic regression model
mdl = fitglm(featureMatrixTraining, trainingLabels, 'Distribution', 'binomial');

% Display the model summary
disp(mdl);

% Predict probabilities for the test data
predictedProbabilities = predict(mdl, featureMatrixTest);

% Convert probabilities to binary labels using a threshold of 0.5
predictedLabels = predictedProbabilities >= 0.5;

% Display the predicted labels
disp(predictedLabels);

comparison = (testingLabels == predictedLabels);
accuracy = sum(comparison) / length(comparison);