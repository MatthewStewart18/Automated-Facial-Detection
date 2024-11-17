clear all
close all
addpath ../../images
addpath ../../utils

% Load training and test data
[trainingImages, trainingLabels] = loadFaceImages('../../images/face_train.cdataset');
[testingImages, testingLabels] = loadFaceImages('../../images/face_test.cdataset');
numTrainingImages = size(trainingImages, 1);
numTestImages = size(testingImages, 1);
fprintf('Loaded augmented training set: %d images\n', size(trainingImages,1));

trainingEdges = extractEdges(trainingImages);

% Train model
fprintf('Training KNN model on entire image ...\n')
modelKNN = NNtraining(trainingEdges, trainingLabels);

% Set K to sqrt(N)
% K = round(sqrt(numTrainingImages)); 
% testEdges = extractEdges(testingImages);
% 
% fprintf('Getting model predictions for K = %d\n', K);
% predictions = zeros(numTestImages);
% for i = 1:numTestImages
%     testEdge = testEdges(i, :);
%     predictions(i, 1) = KNNTesting(testEdge, modelKNN, K);
% end


% Initialize arrays to store K values and their corresponding accuracies
K_values = 1:2:99; % odd values from 1 to 99
accuracies = zeros(length(K_values), 1);

fprintf('Evaluating model for different K.\n');

% Evaluate accuracy for each odd K
for i = 1:length(K_values)
    K = K_values(i); % current value of K
    testEdges = extractEdges(testingImages);
    classificationResult = zeros(numTestImages, 1);

    % For each testing image, obtain a prediction based on the trained model
    for j = 1:numTestImages
        testImage = testEdges(j, :);
        classificationResult(j, 1) = KNNTesting(testImage, modelKNN, K);
    end

    % Calculate accuracy
    comparison = (testingLabels == classificationResult);
    accuracies(i) = sum(comparison) / length(comparison);
end 

highestAcc = round(max(accuracies)*100);
fprintf('Evaluation complete, highest accuracy was %d percent.\n', highestAcc);


% Fit a polynomial curve to show the change in accuracy more clearly
poly_degree = 2;
p = polyfit(K_values, accuracies, poly_degree); % Get polynomial coefficients
fitted_curve = polyval(p, K_values); % Evaluate polynomial at each K

% Plot the fitted curve
figure
hold on;
plot(K_values, fitted_curve, '-r', 'LineWidth', 1.5, 'DisplayName', 'Polynomial Fit');
xlabel('Value of K');
ylabel('Accuracy');
title('KNN Accuracy for Different K');
legend;
grid on;
hold off;




% fprintf('Evaluating model predictions...\n');
% [accuracy, precision, recall, f1_score, confusionMatrix] = calculateMetrics(predictions, testingLabels);
% 
% createRatioBarChartSVM(confusionMatrix, 'KNN', accuracy, precision, recall, f1_score);
% 
% % Display images of the correct/incorrect predictions
% dispPreds(predictions(:, 1), testingLabels, testingImages);
% 
% % Save the trained KNN model to a .mat file
% save('modelKNN.mat', 'modelKNN', 'K');
