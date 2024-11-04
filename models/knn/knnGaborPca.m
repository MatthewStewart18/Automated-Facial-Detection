% Load training and test data
[trainingImages, trainingLabels] = loadFaceImages('../../images/face_train.cdataset');
[testingImages, testingLabels] = loadFaceImages('../../images/face_test.cdataset');
numTrainingImages = size(trainingImages, 1);
numTestImages = size(testingImages, 1);
fprintf('Loaded augmented training set: %d images\n', size(trainingImages,1));

% Set correct image dimensions
imgHeight = 27;
imgWidth = 18;

% Extract Gabor features for training set
fprintf('Extracting Gabor features...\n');
trainingFeatureSet = zeros(size(trainingImages, 1), 19440);

for i = 1:size(trainingImages, 1)
    img = reshape(trainingImages(i,:), [imgHeight, imgWidth]);
    
    % Extract Gabor features
    features = gabor_feature_vector(img);
    
    % Handle any NaN or Inf values
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
    
    trainingFeatureSet(i, :) = features;
end

% Normalize features before PCA
trainingFeatureSet = normalize(trainingFeatureSet, 'zscore');

% Apply PCA with variance retention to training set
[~, score, latent] = pca(trainingFeatureSet);
explained = cumsum(latent)./sum(latent);
n_components = find(explained >= 0.95, 1); % Keep 95% of variance
fprintf('Using %d PCA components\n', n_components);
trainingFeatureSet = score(:, 1:n_components);

% Train model
fprintf('Training KNN model on PCA data ...\n')
modelKNN = NNtraining(trainingFeatureSet, trainingLabels);

% Extract Gabor features for test set
fprintf('Extracting Gabor features...\n');
testingFeatureSet = zeros(size(testingImages, 1), 19440);

for i = 1:numTestImages
    img = reshape(testingImages(i,:), [imgHeight, imgWidth]);
    
    % Extract Gabor features
    features = gabor_feature_vector(img);
    
    % Handle any NaN or Inf values
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
    
    testingFeatureSet(i, :) = features;
end

% Normalize testing features
testingFeatureSet = normalize(testingFeatureSet, 'zscore');

% Apply PCA of same dimension to testing set
[~, score, latent] = pca(testingFeatureSet);
testingFeatureSet = score(:, 1:n_components);

% Set K to sqrt(N)
K = round(sqrt(numTrainingImages));

fprintf('Getting model predictions for K = %d\n', K);
predictions = zeros(numTestImages);
for i = 1:numTestImages
    testImage = testingFeatureSet(i, :);
    predictions(i, 1) = KNNTesting(testImage, modelKNN, K);
end

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictions, testingLabels);