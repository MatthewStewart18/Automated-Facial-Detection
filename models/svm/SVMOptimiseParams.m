clear all
close all

% Add all subdirectories under the base directory
baseDir = '../../';
addpath(genpath(baseDir));

% Load training and test data
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset');
[test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset');

% Define model-specific functions
trainingFunc = @SVMtraining;
predictionFunc = @extractPredictionsSVM;
featureExtractorFunc = @extractEdges;

% Add preprocessing step function
preprocessingFunc = @histEq;

% Range for kerneloption
poly_degrees = 3:0.05:4;

% Storage for performance metrics
accuracy_values = [];
f1_scores = [];
precision_values = [];
recall_values = [];

% Loop over each kerneloption value
for degree = poly_degrees
    % Set the parameter for the current kerneloption
    params = struct('kerneloption', degree, 'kernel', 'polyhomog');
    
    % Create the model object
    svmModel = ModelFactory(trainingFunc, predictionFunc, params, featureExtractorFunc);
    
    % Add preprocessing steps
    svmModel = svmModel.addPreprocessingStep(preprocessingFunc);
    
    % Train the model
    svmModel = svmModel.train(train_images, train_labels);
    
    % Test the model
    predictions = svmModel.test(test_images);
    
    % Calculate metrics
    [accuracy, precision, recall, f1_score, ~] = calculateMetrics(predictions, test_labels);

    % Store the metrics
    accuracy_values = [accuracy_values; accuracy];
    f1_scores = [f1_scores; f1_score];
    precision_values = [precision_values; precision];
    recall_values = [recall_values; recall];
end

% Plot the performance metrics
figure;

% Plot Accuracy
subplot(2, 2, 1);
plot(poly_degrees, accuracy_values, '-ro', 'LineWidth', 1.5);
xlabel('Polynomial Degree');
ylabel('Accuracy');
title('Accuracy vs Polynomial Degree');
grid on;

% Plot F1-Score
subplot(2, 2, 2);
plot(poly_degrees, f1_scores, '-bo', 'LineWidth', 1.5);
xlabel('Polynomial Degree');
ylabel('F1-Score');
title('F1-Score vs Polynomial Degree');
grid on;

% Plot Precision
subplot(2, 2, 3);
plot(poly_degrees, precision_values, '-go', 'LineWidth', 1.5);
xlabel('Polynomial Degree');
ylabel('Precision');
title('Precision vs Polynomial Degree');
grid on;

% Plot Recall
subplot(2, 2, 4);
plot(poly_degrees, recall_values, '-ko', 'LineWidth', 1.5);
xlabel('Polynomial Degree');
ylabel('Recall');
title('Recall vs Polynomial Degree');
grid on;

% Save the results
save('saved-models/kernel-options/poly-hom/metricsSVMEdges.mat', ...
    'poly_degrees', 'accuracy_values', 'f1_scores', 'precision_values', 'recall_values');
