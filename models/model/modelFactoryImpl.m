clear all
close all

% Add relevant files to WD
addpath ../svm
addpath ../knn
addpath ../svm/SVM-KM
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils
addpath ../../feature-extraction-utils/feature-extractors
addpath ../../preprocessing-utils

% Load training and test data
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset');
[test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset', -1);

% Set current model and feature configurations
modelType = ModelType.SVM;
featureType = FeatureType.EdgesPCA;
preprocessingType = PreprocessingType.HistEq;

% Define model parameters
switch modelType
    case ModelType.SVM
        params = struct('kerneloption', 4, 'kernel', 'polyhomog');
    case ModelType.KNN
        params = struct('K', 5);
    case ModelType.LG
        params = {};
        train_labels(train_labels == -1) = 0;
        test_labels(test_labels == -1) = 0;
    case ModelType.RF
        numTrees = 350;
        params = {};
    otherwise
        error('Unsupported ModelType');
end

% Create the model object
if modelType == ModelType.RF
    model = ModelFactory(modelType, featureType, preprocessingType, params, numTrees);
else 
    model = ModelFactory(modelType, featureType, preprocessingType, params);
end

% Train the model
model = model.train(train_images, train_labels);

% Test the model with Test-Time Augmentation (TTA)
final_predictions = zeros(size(test_labels));
final_confidence = zeros(size(test_labels));

% Loop over each test image
for img_idx = 1:size(test_images, 1)
    % Get the current test image
    image = test_images(img_idx, :);
    
    % Apply augmentations to the image
    augmented_images = augmentData(image, test_labels(img_idx), [27, 18]);
    
    % Predict on augmented images
    [predictions, confidence] = model.test(augmented_images);
    
    % Majority voting: select the most common prediction
    final_predictions(img_idx) = mode(predictions);

    % Aggregated Confidence
    if modelType == ModelType.RF
        final_confidence(img_idx) = mean(confidence(:, 2));
    else 
        final_confidence(img_idx) = mean(confidence);
    end
end

% Evaluate the model
[~, ~] = model.evaluate(final_predictions, test_labels, test_images);

% Plot ROC curve
rocCurve(test_labels, final_confidence);

% Save the trained model
savePath = sprintf('saved-models/%s/%s_%s_Model.mat', ...
    char(modelType), char(featureType), char(preprocessingType));

% save(savePath, 'model');
