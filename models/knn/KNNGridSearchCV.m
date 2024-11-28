clear all
close all

% Add relevant files to WD
addpath ../model
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

% K values for KNN to hypertune
K_values = [1, 3, 5, 7, 9, 11, 13, 15];

% Create stratified K-fold partitions on the training data
rng(1);
Kfold = 5;
cv = cvpartition(train_labels, 'KFold', Kfold, 'Stratify', true);

% Array to store performance results for each hyperparameter setting
cv_results = zeros(length(K_values), Kfold);

% Grid search loop
for i = 1:length(K_values)
    K = K_values(i);  % Current hyperparameter value
    
    % Cross-validation loop
    for fold = 1:Kfold
        % Get training and validation indices for the current fold
        train_idx = training(cv, fold);
        val_idx = test(cv, fold);
        
        % Prepare fold-specific training and validation data
        fold_train_images = all_images(train_idx, :);
        fold_train_labels = all_labels(train_idx);
        fold_val_images = all_images(val_idx, :);
        fold_val_labels = all_labels(val_idx);

        % Train the model
        params = struct('K', K);
        [fold_train_images, fold_train_labels] = augmentData(fold_train_images, fold_train_labels, [27, 18]);
        model = ModelFactory(modelType, featureType, preprocessingType, params);
        model = model.train(fold_train_images, fold_train_labels);
        
        % Test the model
        [predictions, confidence] = model.test(fold_val_images);
        
        % Calculate accuracy for this fold
        accuracy = sum(predictions == fold_val_labels) / length(fold_val_labels);
        
        % Store the result
        cv_results(i, fold) = accuracy;
    end
end

% Calculate the average accuracy for each hyperparameter setting
avg_accuracy = mean(cv_results, 2);

% Identify the best K (with the highest average accuracy)
[~, best_K_idx] = max(avg_accuracy);
best_K = K_values(best_K_idx);

fprintf('Best K: %d with average accuracy: %.4f\n', best_K, avg_accuracy(best_K_idx));