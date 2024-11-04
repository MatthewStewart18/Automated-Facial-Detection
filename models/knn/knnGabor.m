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

for i = 1:numTrainingImages
    img = reshape(trainingImages(i,:), [imgHeight, imgWidth]);
    
    % Extract Gabor features
    features = gabor_feature_vector(img);
    
    % Handle any NaN or Inf values
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
    
    trainingFeatureSet(i, :) = features;
end

% Normalize training features
trainingFeatureSet = normalize(trainingFeatureSet, 'zscore');

% Train model
fprintf('Training KNN model on Gabor features ...\n')
modelKNN = NNtraining(trainingFeatureSet, trainingLabels);

% Set K to sqrt(N)
K = round(sqrt(numTrainingImages));

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

fprintf('Getting model predictions for K = %d\n', K);
predictions = zeros(numTestImages);
for i = 1:numTestImages
    testImage = testingFeatureSet(i, :);
    predictions(i, 1) = KNNTesting(testImage, modelKNN, K);
end

fprintf('Evaluating model predictions...\n');
[~] = calculateMetrics(predictions, testingLabels);