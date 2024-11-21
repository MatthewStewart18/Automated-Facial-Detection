clear all
close all
addpath ../../images
addpath ../../utils

% Load training and testing datasets
[trainingImages, trainingLabels] = loadFaceImages("../../images/face_train.cdataset", 1);
[testingImages, testingLabels] = loadFaceImages("../../images/face_test.cdataset", 1);
numTrainingImages = size(trainingImages, 1);
numTestImages = size(testingImages, 1);


% Normalize features before PCA
trainingImagesNorm = normalize(trainingImages, 'zscore');

% Normalize features before PCA
testingImagesNorm = normalize(testingImages, 'zscore');

% Apply PCA with variance retention to training set
[~, score, latent] = pca(trainingImagesNorm);
explained = cumsum(latent)./sum(latent);
n_components = find(explained >= 0.95, 1); % Keep 95% of variance
fprintf('Using %d PCA components\n', n_components);
trainingFeatureSet = score(:, 1:n_components);

% Apply PCA with variance retention to testing set
[~, score, latent] = pca(testingImagesNorm);
explained = cumsum(latent)./sum(latent);
n_components = find(explained >= 0.95, 1); % Keep 95% of variance
fprintf('Using %d PCA components\n', n_components);
testingFeatureSet = score(:, 1:n_components);

% Get the number of columns in testingFeatureSet
numTestingCols = size(testingFeatureSet, 2);

if size(trainingFeatureSet, 2) > numTestingCols
   trainingFeatureSet = trainingFeatureSet(:, 1:numTestingCols);
end

%converting -1 to 0
trainingLabels(trainingLabels == -1) = 0;

testingLabels(testingLabels == -1) = 0;

% Fit logistic regression model
mdl = fitglm(trainingFeatureSet, trainingLabels, 'Distribution', 'binomial');

% Predict probabilities for the test data
predictedProbabilities = predict(mdl, testingFeatureSet);

% Convert probabilities to binary labels using a threshold of 0.5
predictedLabels = double(predictedProbabilities >= 0.5);

%converting 0 to -1
predictedLabels(predictedLabels == 0) = -1;
testingLabels(testingLabels == 0) = -1;

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictedLabels, testingLabels);