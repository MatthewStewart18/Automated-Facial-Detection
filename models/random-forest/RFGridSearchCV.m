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
[images, labels] = loadFaceImages('../../images/face_train.cdataset', -1);

% Set current model and feature configurations
modelType = ModelType.RF;
featureType = FeatureType.HOG;
preprocessingType = PreprocessingType.HistEq;

% K values e
NumTrees_values = 200:25:400;

% Create stratified K-fold partitions on the training data
rng(45);
Kfold = 5;
cv = cvpartition(labels, 'KFold', Kfold, 'Stratify', true);

% Array to store performance results for each hyperparameter setting
cv_results = zeros(length(NumTrees_values), Kfold);
metrics_results = struct('Precision', [], 'Recall', [], 'F1', [], 'ConfusionMatrix', []);

% Grid search loop
for i = 1:length(NumTrees_values)
    numTrees = NumTrees_values(i);  % Current hyperparameter value
    
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
        params = {};
        [train_images, train_labels] = augmentData(train_images, train_labels, [27, 18]);
        model = ModelFactory(modelType, featureType, preprocessingType, params, numTrees);
        model = model.train(train_images, train_labels);
        
        % Test the model
        [predictions, confidence] = model.test(test_images);
        
        % Calculate accuracy and metrics for this fold
        [accuracy, precision, recall, f1_score, confusionMatrix] = calculateMetrics(predictions, test_labels);
        
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
[~, best_tree_idx] = max(avg_accuracy);
best_numTrees = NumTrees_values(best_tree_idx);

% Extract and display the best metrics
best_precision = mean(metrics_results(best_tree_idx).Precision);
best_recall = mean(metrics_results(best_tree_idx).Recall);
best_f1_score = mean(metrics_results(best_tree_idx).F1);
best_confusion_matrix = round(mean(metrics_results(best_tree_idx).ConfusionMatrix, 1));

fprintf('\nBest Num Trees: %d with average accuracy: %.4f%%\n', best_numTrees, avg_accuracy(best_tree_idx));
fprintf('Precision: %.4f\n', best_precision);
fprintf('Recall: %.4f\n', best_recall);
fprintf('F1 Score: %.4f\n', best_f1_score);
fprintf('Confusion Matrix (TP, FP, FN, TN): [%d, %d, %d, %d]\n', best_confusion_matrix);

% Plot the average accuracy for each K value
figure;
plot(NumTrees_values, avg_accuracy, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of Trees');
ylabel('Average Accuracy');
title(sprintf('%s %s Hyperparameter Tuning', modelType, featureType));
grid on;
xticks(NumTrees_values);
xlim([min(NumTrees_values), max(NumTrees_values)]);
ylim([min(avg_accuracy)-0.01, max(avg_accuracy)+0.01]);

% Annotate the best K on the graph
hold on;
best_tree_value = NumTrees_values(best_tree_idx);
best_acc = avg_accuracy(best_tree_idx);
plot(best_tree_value, best_acc, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
text(best_tree_value, best_acc, sprintf(' Best NumTrees=%d, Accuracy=%.4f', best_tree_value, best_acc), ...
     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');