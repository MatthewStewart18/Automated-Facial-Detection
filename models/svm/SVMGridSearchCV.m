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
modelType = ModelType.SVM;
featureType = FeatureType.EdgesPCA;
preprocessingType = PreprocessingType.HistEq;

% kernel options range
kernel_options = 1:5;

% Create stratified K-fold partitions on the training data
rng(45);
Kfold = 5;
cv = cvpartition(labels, 'KFold', Kfold, 'Stratify', true);

% Array to store performance results for each hyperparameter setting
cv_results = zeros(length(kernel_options), Kfold);
metrics_results = struct('Precision', [], 'Recall', [], 'F1', [], 'ConfusionMatrix', []);

% Grid search loop
for i = 1:length(kernel_options)
    degree = kernel_options(i);  % Current hyperparameter value
    
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
        params = struct('kerneloption', degree, 'kernel', 'poly');
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
[~, best_degree_value] = max(avg_accuracy);
best_degree = kernel_options(best_degree_value);

% Extract and display the best metrics
best_precision = mean(metrics_results(best_degree_value).Precision);
best_recall = mean(metrics_results(best_degree_value).Recall);
best_f1_score = mean(metrics_results(best_degree_value).F1);
best_confusion_matrix = round(mean(metrics_results(best_degree_value).ConfusionMatrix, 1));

fprintf('\nBest degree: %d with average accuracy: %.4f%%\n', best_degree, avg_accuracy(best_degree_value));
fprintf('Precision: %.4f\n', best_precision);
fprintf('Recall: %.4f\n', best_recall);
fprintf('F1 Score: %.4f\n', best_f1_score);
fprintf('Confusion Matrix (TP, FP, FN, TN): [%d, %d, %d, %d]\n', best_confusion_matrix);

% Plot the average accuracy for each degree value
figure;
plot(kernel_options, avg_accuracy, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Polynomial Degree');
ylabel('Average Accuracy');
title(sprintf('%s Polynomial Kernel %s Hyperparameter Tuning', modelType, featureType));
grid on;
xticks(kernel_options);
xlim([min(kernel_options), max(kernel_options)]);
ylim([min(avg_accuracy)-0.01, max(avg_accuracy)+0.1]);

% Annotate the best K on the graph
hold on;
best_degree_value = kernel_options(best_degree_value);
best_acc = avg_accuracy(best_degree_value);
plot(best_degree_value, best_acc, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
text(best_degree_value, best_acc, sprintf(' Best degree=%d, Accuracy=%.4f', best_degree_value, best_acc), ...
     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');