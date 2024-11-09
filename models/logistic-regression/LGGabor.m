clear all
close all
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils/gabor

% Load training and testing datasets
[trainingImages, trainingLabels] = loadFaceImages("../../images/face_train.cdataset", 1);
[testingImages, testingLabels] = loadFaceImages("../../images/face_test.cdataset", 1);
numTrainingImages = size(trainingImages, 1);
numTestImages = size(testingImages, 1);

featureMatrixTraining = zeros(numTrainingImages, 19440);
featureMatrixTest = zeros(numTestImages, 19440);

%reshaping to 3d image
imageListTraining = reshape2dImage(trainingImages);

for imageNumTraining = 1:length(imageListTraining)

    image = cell2mat(imageListTraining(imageNumTraining));
    image = normalize(image, 'zscore');
    featureMatrixTraining(imageNumTraining, :) = gabor_feature_vector(image);
     % Handle any NaN or Inf values
 % Handle any NaN or Inf values for the specific row only
featureMatrixTraining(imageNumTraining, isnan(featureMatrixTraining(imageNumTraining, :))) = 0;
featureMatrixTraining(imageNumTraining, isinf(featureMatrixTraining(imageNumTraining, :))) = 0;
end

%reshaping to 3d image
imageListTest = reshape2dImage(testingImages);
for imageNumTest = 1:length(imageListTest)
    image = cell2mat(imageListTest(imageNumTest));
    image = normalize(image, 'zscore');
    featureMatrixTest(imageNumTest, :) = gabor_feature_vector(image);

% Handle any NaN or Inf values for the specific row only
featureMatrixTest(imageNumTest, isnan(featureMatrixTest(imageNumTest, :))) = 0;
featureMatrixTest(imageNumTest, isinf(featureMatrixTest(imageNumTest, :))) = 0;
end

%converting -1 to 0
trainingLabels(trainingLabels == -1) = 0;

testingLabels(testingLabels == -1) = 0;

% Fit logistic regression model
mdl = fitglm(featureMatrixTraining, trainingLabels, 'Distribution', 'binomial');

% Predict probabilities for the test data
predictedProbabilities = predict(mdl, featureMatrixTest);

% Convert probabilities to binary labels using a threshold of 0.5
predictedLabels = predictedProbabilities >= 0.5;

% Display the predicted labels
disp(predictedLabels);

%converting 0 to -1
predictedLabels(predictedLabels == 0) = -1;
testingLabels(testingLabels == 0) = -1;

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictedLabels, testingLabels);