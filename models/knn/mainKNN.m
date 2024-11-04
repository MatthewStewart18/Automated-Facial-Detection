clear all
close all
addpath ../../images

% Load training and test data
[trainingImages, trainingLabels] = loadFaceImages('../../images/face_train.cdataset');
[testingImages, testingLabels] = loadFaceImages('../../images/face_test.cdataset');
numTestImages = size(testingImages, 1);

fprintf('Loaded training set: %d images\n', size(trainingImages,1));

% Apply PCA with variance retention to training set
[~, score, latent] = pca(trainingImages);
explained = cumsum(latent)./sum(latent);
n_components = find(explained >= 0.95, 1); % Keep 95% of variance
fprintf('Using %d PCA components\n', n_components);
trainingFeatureSet = score(:, 1:n_components);

% Train model
fprintf('Training KNN model on PCA data ...\n')
modelKNN = NNtraining(trainingFeatureSet, trainingLabels);

% Apply PCA of same dimension to testing set
[~, score, latent] = pca(testingImages);
testingFeatureSet = score(:, 1:n_components);

% Initialize arrays to store K values and their corresponding accuracies
K_values = 1:2:99; % odd values from 1 to 99
accuracies = zeros(length(K_values), 1);

fprintf('Evaluating model for different K.\n');

% Evaluate accuracy for each odd K
for i = 1:length(K_values)
    K = K_values(i); % current value of K
    classificationResult = zeros(numTestImages, 1);

    % For each testing image, obtain a prediction based on the trained model
    for j = 1:numTestImages
        testImage = testingFeatureSet(j, :);
        classificationResult(j, 1) = KNNTesting(testImage, modelKNN, K);
    end

    % Calculate accuracy
    comparison = (testingLabels == classificationResult);
    accuracies(i) = sum(comparison) / length(comparison);
end 

highestAcc = round(max(accuracies)*100);
fprintf('Evaluation complete, highest accuracy was %d percent.\n', highestAcc);


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