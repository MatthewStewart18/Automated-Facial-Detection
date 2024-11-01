clear all
close all
addpath ../../images
addpath ../../preprocessing-utils/power-law
addpath ../../preprocessing-utils/hist-eq
addpath ../../utils

% Load training and testing datasets
[trainingImages, trainingLabels] = loadFaceImages("../../images/face_train.cdataset", 1);
[testingImages, testingLabels] = loadFaceImages("../../images/face_test.cdataset", 1);
numTrainImages = size(trainingImages, 1);
numTestImages = size(testingImages, 1);

% pre-process data-sets
for i = 1:numTrainImages
    trainingImages(i, :) = enhanceContrastHE(uint8(trainingImages(i, :)));
end

for i = 1:numTestImages
    testingImages(i, :) = enhanceContrastHE(uint8(testingImages(i, :)));
end

images = vertcat(trainingImages, testingImages);
[U,S,X_reduce] = pca(images, 2);

% Train model
modelKNN = NNtraining(trainingImages, trainingLabels);

% Initialize arrays to store K values and their corresponding accuracies
K_values = 1:2:99; % odd values from 1 to 99
accuracies = zeros(length(K_values), 1);

% Evaluate accuracy for each odd K
for i = 1:length(K_values)
    K = K_values(i); % current value of K
    classificationResult = zeros(numTestImages, 1);

    % For each testing image, obtain a prediction based on the trained model
    for j = 1:numTestImages
        testImage = testingImages(j, :);
        classificationResult(j, 1) = KNNTesting(testImage, modelKNN, K);
    end

    % Calculate accuracy
    comparison = (testingLabels == classificationResult);
     accuracies(i) = sum(comparison) / length(comparison);
end 

% Fit a polynomial curve to show the change in accuracy more clearly
poly_degree = 5;
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
