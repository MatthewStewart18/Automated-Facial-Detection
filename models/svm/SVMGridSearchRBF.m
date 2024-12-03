clear all
close all

% Add relevant files to WD
addpath SVM-KM
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
featureType = FeatureType.HOG;
preprocessingType = PreprocessingType.HistEq;

% kernel options range
kernel_options = 2.^(-3:2:7);
C_values = 2.^(3:2:7);

kernel_options = {0.25};
kernel_options = cell2mat(kernel_options);
C_values = 8;

% Create stratified K-fold partitions on the training data
rng(45);
Kfold = 5;
cv = cvpartition(labels, 'KFold', Kfold, 'Stratify', true);

% Array to store performance results for each hyperparameter setting
cv_results = zeros(length(kernel_options), Kfold);
metrics_results = struct('Precision', [], 'Recall', [], 'F1', [], 'ConfusionMatrix', []);

% Grid search loop
for i = 1:length(kernel_options)
    gamma = kernel_options(i);  % Current gamma value
    
    for j = 1:length(C_values)
        C = C_values(j);  % Current C value
        
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

            % Train the model with both gamma and C parameters
            params = struct('kerneloption', gamma, 'kernel', 'gaussian', 'C', C);
            [train_images, train_labels] = augmentData(train_images, train_labels, [27, 18]);
            model = ModelFactory(modelType, featureType, preprocessingType, params);
            model = model.train(train_images, train_labels);

            % Test the model with Test-Time Augmentation (TTA)
            final_predictions = zeros(size(test_labels));
            
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
            cv_results(i, j, fold) = accuracy;
            metrics_results(i, j).Precision(fold) = precision;
            metrics_results(i, j).Recall(fold) = recall;
            metrics_results(i, j).F1(fold) = f1_score;
            metrics_results(i, j).ConfusionMatrix(fold, :) = confusionMatrix;
        end
    end
end


% Calculate the average accuracy for each hyperparameter setting
avg_accuracy = mean(cv_results, 3);  % Average across folds
[best_acc, idx] = max(avg_accuracy(:));  % Find the best accuracy
[best_gamma_idx, best_C_idx] = ind2sub(size(avg_accuracy), idx);
best_gamma = kernel_options(best_gamma_idx);
best_C = C_values(best_C_idx);

% Extract and display the best metrics
best_precision = mean(metrics_results(best_gamma_idx, best_C_idx).Precision);
best_recall = mean(metrics_results(best_gamma_idx, best_C_idx).Recall);
best_f1_score = mean(metrics_results(best_gamma_idx, best_C_idx).F1);
best_confusion_matrix = round(mean(metrics_results(best_gamma_idx, best_C_idx).ConfusionMatrix, 1));

fprintf('\nBest gamma: %g, Best C: %g with average accuracy: %.4f%%\n', best_gamma, best_C, best_acc);
fprintf('Precision: %.4f\n', best_precision);
fprintf('Recall: %.4f\n', best_recall);
fprintf('F1 Score: %.4f\n', best_f1_score);
fprintf('Confusion Matrix (TP, FP, FN, TN): [%d, %d, %d, %d]\n', best_confusion_matrix);