clear all
close all
addpath ../../images
addpath ../../preprocessing-utils/power-law
addpath ../../preprocessing-utils/hist-eq
addpath ../../utils

% Load training and testing datasets
[trainingImages, trainingLabels] = loadFaceImages("../../images/face_train.cdataset", 1);
[testingImages, testingLabels] = loadFaceImages("../../images/face_test.cdataset", 1);
images = vertcat(trainingImages, testingImages);
labels = vertcat(trainingLabels, testingLabels);

% Apply PCA
[~,S,X_reduce] = pca(images, 2);

figure;
gscatter(X_reduce(:,1), X_reduce(:,2), labels, 'rb', 'ox');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('2D PCA Scatter Plot of Face and Non-Face Images');
legend('Non-Faces', 'Faces');

corr2D = corr(X_reduce);
avgCorr2D = mean(corr2D(:));

% Get the eigenvalues from the diagonal of S
eigenvalues = diag(S);

total_variance = sum(eigenvalues);
explained_variance = eigenvalues / total_variance;
cumulative_explained_variance = cumsum(explained_variance);

% Plot the cumulative explained variance
figure;
plot(cumulative_explained_variance, 'b-', 'LineWidth', 2);
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance');
title('Cumulative Explained Variance by Principal Components');
grid on;

hold on;
yline(0.95, 'r--', 'LineWidth', 1.5, 'Label', '95% Threshold', 'LabelVerticalAlignment', 'bottom');
num_components_95 = find(cumulative_explained_variance >= 0.95, 1);
xline(num_components_95, 'k--', 'LineWidth', 1.5, 'Label', sprintf('%d Components', num_components_95), 'LabelHorizontalAlignment', 'right');
hold off;












% numTrainImages = size(trainingImages, 1);
% numTestImages = size(testingImages, 1);
% % pre-process data-sets
% for i = 1:numTrainImages
%     trainingImages(i, :) = enhanceContrastPL(uint8(trainingImages(i, :)), 0.5);
% end
% 
% for i = 1:numTestImages
%     testingImages(i, :) = enhanceContrastPL(uint8(testingImages(i, :)), 0.5);
% end
% 
% images = vertcat(trainingImages, testingImages);
% [U,S,X_reduce] = pca(images, 2);
% 
% figure;
% gscatter(X_reduce(:,1), X_reduce(:,2), labels, 'rb', 'ox');
% xlabel('Principal Component 1');
% ylabel('Principal Component 2');
% title('2D PCA Scatter Plot after Histogram Eq');
% legend('Non-Faces', 'Faces');
% 
% corrAfter = corr(X_reduce);
% 
% avg_corr_before = mean(corrBefore(:));
% avg_corr_after = mean(corrAfter(:));
% 
% fprintf('Average correlation before hist eq: %.3f\n', avg_corr_before);
% fprintf('Average correlation after hist eq: %.3f\n', avg_corr_after);














% % Train model
% modelKNN = NNtraining(trainingImages, trainingLabels);
% 
% % Initialize arrays to store K values and their corresponding accuracies
% K_values = 1:2:99; % odd values from 1 to 99
% accuracies = zeros(length(K_values), 1);
% 
% % Evaluate accuracy for each odd K
% for i = 1:length(K_values)
%     K = K_values(i); % current value of K
%     classificationResult = zeros(numTestImages, 1);
% 
%     % For each testing image, obtain a prediction based on the trained model
%     for j = 1:numTestImages
%         testImage = testingImages(j, :);
%         classificationResult(j, 1) = KNNTesting(testImage, modelKNN, K);
%     end
% 
%     % Calculate accuracy
%     comparison = (testingLabels == classificationResult);
%      accuracies(i) = sum(comparison) / length(comparison);
% end 
% 
% % Fit a polynomial curve to show the change in accuracy more clearly
% poly_degree = 5;
% p = polyfit(K_values, accuracies, poly_degree); % Get polynomial coefficients
% fitted_curve = polyval(p, K_values); % Evaluate polynomial at each K
% 
% % Plot the fitted curve
% figure
% hold on;
% plot(K_values, fitted_curve, '-r', 'LineWidth', 1.5, 'DisplayName', 'Polynomial Fit');
% xlabel('Value of K');
% ylabel('Accuracy');
% title('KNN Accuracy for Different K');
% legend;
% grid on;
% hold off;
