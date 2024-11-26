clear all
close all

% Add relevant files to WD
addpath ../svm
addpath ../knn
addpath ../svm/SVM-KM
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils
addpath ../../feature-extraction-utils/feature-extractors
addpath ../../preprocessing-utils

% Load training and test data without augmentation
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset', -1);
[test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset', -1);

% Concatenate training and test set before cross-validation
all_images = [train_images; test_images]; 
all_labels = [train_labels; test_labels];

% Set current model and feature configurations
modelType = ModelType.KNN;
featureType = FeatureType.EdgesPCA;
preprocessingType = PreprocessingType.HistEq;

% Define model parameters
switch modelType
    case ModelType.SVM
        params = struct('kerneloption', 2.5, 'kernel', 'polyhomog');
    case ModelType.KNN
        params = struct('K', round(sqrt(size(train_labels, 1))));
    case ModelType.LG
        params = {};
        train_labels(train_labels == -1) = 0;
        test_labels(test_labels == -1) = 0;
    otherwise
        error('Unsupported ModelType');
end

N = size(all_images, 1); % Total number of samples
imageSize = [27, 18];

% Array to store metrics
accuracies = zeros(N, 1);

% Leave-One-Out Cross-Validation loop
for i = 1:N
    % Split the data using 'LeaveMOut' with M=1
    [train_idx, test_idx] = crossvalind('LeaveMOut', N, 1);

    % Prepare training and testing data
    train_images = all_images(train_idx, :);
    train_labels = all_labels(train_idx);
    test_images = all_images(test_idx, :);
    test_labels = all_labels(test_idx);
    
    % Augment the images
    [train_images, train_labels] = augmentData(train_images, train_labels, imageSize);
    [test_images, test_labels] = augmentData(test_images, test_labels, imageSize);

    % Create and train the model
    model = ModelFactory(modelType, featureType, preprocessingType, params);
    model = model.train(train_images, train_labels);

    % Test the model
    predictions = model.test(test_images);
    
    % Evaluate model
    [accuracy, ~] = model.evaluate(predictions, test_labels, test_images);
    
    % Store results
    accuracies(i) = accuracy;
end

% Calculate the average accuracy
averageAccuracy = sum(accuracies)/N;

% Print the average accuracy
fprintf('Leave-One-Out Cross-Validation Average Accuracy: %.4f%%\n', averageAccuracy);

% Save the trained model
savePath = sprintf('saved-models/%s/%s_%s_Model.mat', ...
    char(modelType), char(featureType), char(preprocessingType));
% save(savePath, 'model');