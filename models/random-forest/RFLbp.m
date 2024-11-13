clear all;
close all;

% Add required paths
addpath ../../images
addpath ../../utils

% Load training and test data
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset');
[test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset');
numTrainingImages = size(train_images, 1);
numTestImages = size(test_images, 1);
fprintf('Loaded training set: %d images\n', size(train_images,1));

% Set correct image dimensions
img_height = 27;
img_width = 18;

% Extract LBP features
fprintf('Extracting LBP features...\n');
cellSize = [6 6];  % Adjust cell size for more features
numNeighbors = 10; 
radius = 1;  


% Get size of LBP features for one image to initialize matrix
temp_img = reshape(train_images(1,:), [img_height img_width]);
temp_features = extractLBPFeatures(temp_img, 'CellSize', cellSize, ...
    'NumNeighbors', numNeighbors, 'Radius', radius);
feature_length = length(temp_features);

% Initialize feature matrices
train_features = zeros(size(train_images,1), feature_length);
train_images = normalize(train_images, 'zscore');

for i = 1:size(train_images,1)
    if mod(i, 10) == 0
        fprintf('Processing training image %d/%d\n', i, size(train_images,1));
    end
    img = reshape(train_images(i,:), [img_height img_width]);
    
    % Extract LBP features
    features = extractLBPFeatures(img, 'CellSize', cellSize, ...
        'NumNeighbors', numNeighbors, 'Radius', radius);
    
    train_features(i, :) = features;
end
trainingFeatureSet = train_features;

% Normalize features before PCA
trainingFeatureSet = normalize(trainingFeatureSet, 'zscore');

% Apply PCA with variance retention to training set
[~, score, latent] = pca(trainingFeatureSet);
explained = cumsum(latent)./sum(latent);
n_components = 239; % Keep 95% of variance
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
% Extract LBP features from test images
test_features = zeros(size(test_images,1), feature_length);
test_images = normalize(test_images, 'zscore');

for i = 1:size(test_images,1)
    if mod(i, 10) == 0
        fprintf('Processing testing image %d/%d\n', i, size(test_images,1));
    end
    img = reshape(test_images(i,:), [img_height img_width]);
    
    % Extract LBP features
    features = extractLBPFeatures(img, 'CellSize', cellSize, ...
        'NumNeighbors', numNeighbors, 'Radius', radius);
    
    test_features(i, :) = features;
end
testingFeatureSet = test_features;

% Normalize features before PCA
testingFeatureSet = normalize(testingFeatureSet, 'zscore');

% Apply PCA with variance retention to training set
[~, score, latent] = pca(testingFeatureSet);
explained = cumsum(latent)./sum(latent);
fprintf('Using %d PCA components\n', n_components);
testingFeatureSet = score(:, 1:n_components);
testingFeatureSet = normalize(testingFeatureSet, 'zscore');


predictedLabels = predict(rfModel, testingFeatureSet);
predictedLabels = str2double(predictedLabels);
fprintf('Evaluating model predictions...\n');
[accuracy, precision, recall, f1_score, confusionMatrix] = calculateMetrics(predictedLabels, test_labels);

createRatioBarChartSVM(confusionMatrix, "Edges Random-Forest", accuracy, precision, recall,f1_score)
