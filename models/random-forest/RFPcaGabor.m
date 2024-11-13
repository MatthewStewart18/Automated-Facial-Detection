clear all;
close all;

% Add required paths
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils/gabor

% Load training and test data
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset');
[test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset');
numTrainingImages = size(train_images, 1);
numTestImages = size(test_images, 1);
fprintf('Loaded training set: %d images\n', size(train_images,1));

% Set correct image dimensions
img_height = 27;
img_width = 18;

% Extract Gabor features
fprintf('Extracting Gabor features...\n');
train_features = zeros(size(train_images,1), 19440);
train_images = normalize(train_images, 'zscore');
for i = 1:size(train_images,1)
    if mod(i, 10) == 0
        fprintf('Processing training image %d/%d\n', i, size(train_images,1));
    end
    img = reshape(train_images(i,:), [img_height img_width]);
    
    % Extract Gabor features
    features = gabor_feature_vector(img);
    
    % Handle any NaN or Inf values
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
    
    train_features(i, :) = features;
end

% Normalize features before PCA
train_features = normalize(train_features, 'zscore');


% Apply PCA with variance retention to training set
[~, score, latent] = pca(train_features);
explained = cumsum(latent)./sum(latent);
n_components = find(explained >= 0.95, 1); % Keep 95% of variance
fprintf('Using %d PCA components\n', n_components);
trainingFeatureSet = score(:, 1:n_components);
trainingFeatureSet = normalize(trainingFeatureSet, 'zscore');

% Load example data (replace this with your own data)
X = [trainingFeatureSet];   % Example feature matrix with 100 samples and 10 features
Y = [train_labels];  % Example binary response vector

% Define the number of folds (use higher folds like 10 for small datasets)
numFolds = 10; % Or 10 if the dataset is very small
cv = cvpartition(Y, 'KFold', numFolds);

% Number of trees in each model (lower values are usually fine for small data)
numTrees = 350; % Adjust as needed to prevent overfitting

% Array to store metrics
accuracy = zeros(numFolds, 1); % Accuracy per fold

confusionMatrixList = cell(1, numFolds);

for i = 1:numFolds
    % Get indices for training and test sets for each fold
    trainIdx = training(cv, i);
    testIdx = test(cv, i);

    % Split data
    X_train = X(trainIdx, :);
    Y_train = Y(trainIdx);
    X_test = X(testIdx, :);
    Y_test = Y(testIdx);

    % Train Random Forest on the training data
    rfModel = TreeBagger(numTrees, X_train, Y_train, 'Method', 'classification');

    % Predict on the test set
    predictedLabels = predict(rfModel, X_test);
    predictedLabels = str2double(predictedLabels); % Convert cell to numeric

   
    [TP, TN, FP, FN] = getConfusionMatrix(predictedLabels, Y_test);

    metricsStruct = struct('TruePositive', TP, 'TrueNegative', TN, 'FalsePositive', FP, 'FalseNegative', FN);

    confusionMatrixList{i} = metricsStruct;

    % Calculate accuracy for this fold
    accuracy(i) = sum(predictedLabels == Y_test) / length(Y_test);
end

% Calculate and display the average accuracy
avgAccuracy = mean(accuracy);
disp(['Average Cross-Validated Accuracy: ', num2str(avgAccuracy * 100), '%']);


% Normalize the test images before extracting the Gabor features
test_features = zeros(size(test_images,1), 19440);
test_images = normalize(test_images, 'zscore');

for i = 1:size(test_images,1)
    if mod(i, 10) == 0
        fprintf('Processing testing image %d/%d\n', i, size(test_images,1));
    end
    img = reshape(test_images(i,:), [img_height img_width]);
    
    % Extract Gabor features
    features = gabor_feature_vector(img);
    
    % Handle any NaN or Inf values
    features(isnan(features)) = 0;
    features(isinf(features)) = 0;
    
    test_features(i, :) = features;
end

% Normalize features before PCA
test_features = normalize(test_features, 'zscore');


% Apply PCA with variance retention to testing set
[~, score, latent] = pca(test_features);
explained = cumsum(latent)./sum(latent);
fprintf('Using %d PCA components\n', n_components);
testingFeatureSet = score(:, 1:n_components);

testingFeatureSet = normalize(testingFeatureSet, 'zscore');

predictedLabels = predict(rfModel, testingFeatureSet);
predictedLabels = str2double(predictedLabels);
fprintf('Evaluating model predictions...\n');
[accuracy, precision, recall, f1_score, confusionMatrix] = calculateMetrics(predictedLabels, test_labels);

createRatioBarChartSVM(confusionMatrix, "Edges Random-Forest", accuracy, precision, recall,f1_score)
