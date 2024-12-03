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
modelType = ModelType.SVM;
featureType = FeatureType.HOG;
preprocessingType = PreprocessingType.HistEq;

% Define model parameters
switch modelType
    case ModelType.SVM
        params = struct('kerneloption', 0.5, 'kernel', 'gaussian', 'C', 8);
    case ModelType.KNN
        params = struct('K', round(sqrt(size(labels, 1))));
        params = struct('K', 1);
    case ModelType.LG
        params = {};
        labels(labels == -1) = 0;
    case ModelType.RF
        numTrees = 250;
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
misclassified_images_all = []; % To store misclassified images from all folds
misclassified_labels_all = []; % To store labels of misclassified images

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
    if modelType == ModelType.RF
        model = ModelFactory(modelType, featureType, preprocessingType, params, numTrees);
    else
        model = ModelFactory(modelType, featureType, preprocessingType, params);
    end
    model = model.train(train_images, train_labels);

    % Test the model with Test-Time Augmentation (TTA)
    final_predictions = zeros(size(test_labels));
    final_conf = zeros(size(test_labels));
    misclassified_indices = []; % Store indices of misclassified samples for this fold
    
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
        
        % Check for misclassification
        if final_predictions(img_idx) ~= test_labels(img_idx)
            misclassified_indices = [misclassified_indices; img_idx];
        end
    end
    
    % Store true labels and confidences for overall ROC
    all_true_labels = [all_true_labels; test_labels];
    all_confidences = [all_confidences; final_conf];
    
    % Save misclassified images and labels from this fold
    misclassified_images_all = [misclassified_images_all; test_images(misclassified_indices, :)];
    misclassified_labels_all = [misclassified_labels_all; test_labels(misclassified_indices)];
    
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
fprintf('Average Accuracy: %.4f%%\n', avg_accuracy);
fprintf('Precision: %.4f\n', avg_precision);
fprintf('Recall: %.4f\n', avg_recall);
fprintf('F1 Score: %.4f\n', avg_f1_score);
fprintf('Confusion Matrix (TP, FP, FN, TN): [%d, %d, %d, %d]\n', avg_confusion_matrix);

% Display all misclassified images in a single figure
num_misclassified = size(misclassified_images_all, 1);
figure;
rows = ceil(sqrt(num_misclassified));
cols = ceil(num_misclassified / rows);
for idx = 1:num_misclassified
    img = reshape(misclassified_images_all(idx, :), [27, 18]); % Assuming 27x18 image size
    subplot(rows, cols, idx);
    imshow(img, []);
    title(sprintf('Label: %d, Conf: %.2f', misclassified_labels_all(idx), final_conf(idx)));
end
sgtitle('All Misclassified Images'); % Add a supertitle to the figure

% Plot overall ROC curve
rocCurve(all_true_labels, all_confidences);
