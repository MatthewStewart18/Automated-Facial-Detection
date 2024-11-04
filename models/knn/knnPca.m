clear all
close all
addpath ../../images

% Load training and test data
[trainingImages, trainingLabels] = loadFaceImages('../../images/face_train.cdataset');
[testingImages, testingLabels] = loadFaceImages('../../images/face_test.cdataset');
numTrainingImages = size(trainingImages, 1);
numTestImages = size(testingImages, 1);

fprintf('Loaded augmented training set: %d images\n', size(trainingImages,1));

% Apply PCA with variance retention to training set
[~, score, latent] = pca(trainingImages);
explained = cumsum(latent)./sum(latent);
n_components = find(explained >= 0.95, 1); % Keep 95% of variance
fprintf('Using %d PCA components\n', n_components);
trainingFeatureSet = score(:, 1:n_components);

% Train model
fprintf('Training KNN model on PCA data ...\n')
modelKNN = NNtraining(trainingFeatureSet, trainingLabels);

% Apply PCA of same dimension to testing set
[~, score, latent] = pca(testingImages);
testingFeatureSet = score(:, 1:n_components);

% Set K to sqrt(N)
K = round(sqrt(numTrainingImages)); 
% this isnt working great, using 50 temporarily
% K = 50;

fprintf('Getting model predictions for K = %d\n', K);
predictions = zeros(numTestImages);
for i = 1:numTestImages
    testImage = testingFeatureSet(i, :);
    predictions(i, 1) = KNNTesting(testImage, modelKNN, K);
end

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictions, testingLabels);