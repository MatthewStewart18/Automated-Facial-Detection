clear all
close all

% Add relevant files to WD
addpath ../model
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils
addpath ../../feature-extraction-utils/feature-extractors
addpath ../../preprocessing-utils

% Load training data without augmentation
[Xtrain, Ytrain] = loadFaceImages('../../images/face_train.cdataset', -1);
[Xtest, Ytest] = loadFaceImages('../../images/face_test.cdataset', -1);

% Combine images and labels
images = [Xtrain; Xtest];
labels = [Ytrain; Ytest];

% Set current model and feature configurations
modelType = ModelType.RF;
featureType = FeatureType.HOG;
preprocessingType = PreprocessingType.HistEq;

% Define model parameters
switch modelType
    case ModelType.SVM
        params = struct('kerneloption', 2, 'kernel', 'polyhomog');
    case ModelType.KNN
        params = struct('K', round(sqrt(size(train_labels, 1))));
    case ModelType.LG
        params = {};
        train_labels(train_labels == -1) = 0;
        test_labels(test_labels == -1) = 0;
    case ModelType.RF
        numTrees = 275;
        params = {};
    otherwise
        error('Unsupported ModelType');
end

% Create stratified K-fold partitions on the training data
rng(45);
Kfold = 5;
cv = cvpartition(labels, 'KFold', Kfold, 'Stratify', true);

% Arrays to store performance results and for overall ROC
all_confidences = [];
all_true_labels = [];

% Cross-validation loop
for fold = 1:Kfold
    % Get training and validation indices for the current fold
    train_idx = training(cv, fold);
    test_idx = test(cv, fold);
    
    % Prepare fold-specific training and validation data
    train_images = images(train_idx, :);
    train_labels = labels(train_idx);
    test_images = images(test_idx, :);
    test_labels = labels(test_idx);

    % Train the model
    [train_images, train_labels] = augmentData(train_images, train_labels, [27, 18]);
    model = ModelFactory(modelType, featureType, preprocessingType, params, numTrees);
    model = model.train(train_images, train_labels);

    % Test the model with Test-Time Augmentation (TTA)
    final_predictions = zeros(size(test_labels));
    final_conf = zeros(size(test_labels));
    
    % Loop over each test image
    for img_idx = 1:size(test_images, 1)
        image = test_images(img_idx, :);
        augmented_images = augmentData(image, test_labels(img_idx), [27, 18]);
        
        % Predict on augmented images
        [predictions, confidence] = model.test(augmented_images);
        
        % Majority voting
        final_predictions(img_idx) = mode(predictions);
        
        % Aggregated confidence
        if modelType == ModelType.RF
            final_conf(img_idx) = mean(confidence(:, 2));
        else
            final_conf(img_idx) = mean(confidence);
        end
    end
    
    % Store true labels and confidences for overall ROC
    all_true_labels = [all_true_labels; test_labels];
    all_confidences = [all_confidences; final_conf];
    
    % Calculate accuracy and metrics for this fold
    [accuracy, precision, recall, f1_score, confusionMatrix] = calculateMetrics(final_predictions, test_labels);
    
    % Store results
    cv_results(fold) = accuracy;
    metrics_results.Precision(fold) = precision;
    metrics_results.Recall(fold) = recall;
    metrics_results.F1(fold) = f1_score;
    metrics_results.ConfusionMatrix(fold, :) = confusionMatrix;
end

% Calculate average metrics across folds
avg_accuracy = mean(cv_results);
avg_precision = mean(metrics_results.Precision);
avg_recall = mean(metrics_results.Recall);
avg_f1_score = mean(metrics_results.F1);
avg_confusion_matrix = round(mean(metrics_results.ConfusionMatrix, 1));

% Display results
fprintf('\nResults for %d Trees:\n', numTrees);
fprintf('Average Accuracy: %.4f%%\n', avg_accuracy);
fprintf('Precision: %.4f\n', avg_precision);
fprintf('Recall: %.4f\n', avg_recall);
fprintf('F1 Score: %.4f\n', avg_f1_score);
fprintf('Confusion Matrix (TP, FP, FN, TN): [%d, %d, %d, %d]\n', avg_confusion_matrix);

% Plot overall ROC curve
rocCurve(all_true_labels, all_confidences);
