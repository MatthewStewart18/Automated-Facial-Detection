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

% Concatenate images vertically (rows correspond to images)
images = [Xtrain; Xtest];
labels = [Ytrain; Ytest];

% Set current model and feature configurations
modelType = ModelType.KNN;
featureType = FeatureType.LBP;
preprocessingType = PreprocessingType.HistEq;

% K values e
K_values = 9:2:49;

% Create stratified K-fold partitions on the training data
rng(45);
Kfold = 5;
cv = cvpartition(labels, 'KFold', Kfold, 'Stratify', true);

% Array to store performance results for each hyperparameter setting
cv_results = zeros(length(K_values), Kfold);
metrics_results = struct('Precision', [], 'Recall', [], 'F1', [], 'ConfusionMatrix', []);

% Grid search loop
for i = 1:length(K_values)
    K = K_values(i);  % Current hyperparameter value
    
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
        params = struct('K', K);
        [train_images, train_labels] = augmentData(train_images, train_labels, [27, 18]);
        model = ModelFactory(modelType, featureType, preprocessingType, params);
        model = model.train(train_images, train_labels);

        % Test the model with Test-Time Augmentation (TTA)
        final_predictions = zeros(size(test_labels));
        final_conf = zeros(size(test_labels));
        
        % Loop over each test image
        for img_idx = 1:size(test_images, 1)
            % Get the current test image
            image = test_images(img_idx, :);
            
            % Apply augmentations to the image
            augmented_images = augmentData(image, test_labels(img_idx), [27, 18]);
            
            % Predict on augmented images
            [predictions, ~] = model.test(augmented_images);
            
            % Majority voting: select the most common prediction
            final_predictions(img_idx) = mode(predictions);
        end
        
        % Calculate accuracy and metrics for this fold
        [accuracy, precision, recall, f1_score, confusionMatrix] = calculateMetrics(final_predictions, test_labels);
        
        % Store the accuracy and other metrics
        cv_results(i, fold) = accuracy;
        metrics_results(i).Precision(fold) = precision;
        metrics_results(i).Recall(fold) = recall;
        metrics_results(i).F1(fold) = f1_score;
        metrics_results(i).ConfusionMatrix(fold, :) = confusionMatrix;
    end
end

% Calculate the average accuracy for each hyperparameter setting
avg_accuracy = mean(cv_results, 2);

% Filter to find the best K above a certain threshold (e.g., K >= 5)
[~, best_K_value] = max(avg_accuracy);
best_K = K_values(best_K_value);

% Extract and display the best metrics
best_precision = mean(metrics_results(best_K_value).Precision);
best_recall = mean(metrics_results(best_K_value).Recall);
best_f1_score = mean(metrics_results(best_K_value).F1);
best_confusion_matrix = round(mean(metrics_results(best_K_value).ConfusionMatrix, 1));

fprintf('\nBest k-value: %d with average accuracy: %.4f%%\n', best_K, avg_accuracy(best_K_value));
fprintf('Precision: %.4f\n', best_precision);
fprintf('Recall: %.4f\n', best_recall);
fprintf('F1 Score: %.4f\n', best_f1_score);
fprintf('Confusion Matrix (TP, FP, FN, TN): [%d, %d, %d, %d]\n', best_confusion_matrix);

% Plot the average accuracy for each K value
figure;
plot(K_values, avg_accuracy, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Value of K');
ylabel('Average Accuracy');
title(sprintf('%s %s Hyperparameter Tuning', modelType, featureType));
grid on;
xticks(K_values);
xlim([min(K_values), max(K_values)]);
ylim([min(avg_accuracy)-0.01, max(avg_accuracy)+0.01]);

% Annotate the best K on the graph
hold on;
best_k_value = K_values(best_K_value);
best_acc = avg_accuracy(best_K_value);
plot(best_k_value, best_acc, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
text(best_k_value, best_acc, sprintf(' Best K=%d, Accuracy=%.4f', best_k_value, best_acc), ...
     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');