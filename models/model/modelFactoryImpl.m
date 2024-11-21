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
[test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset');

% Set current model and feature configurations
modelType = ModelType.SVM;
featureType = FeatureType.Edges;
preprocessingType = PreprocessingType.None;

% Define model parameters
switch modelType
    case ModelType.SVM
        params = struct('kerneloption', 3.85, 'kernel', 'polyhomog');
    case ModelType.KNN
        params = struct('K', round(sqrt(size(train_labels, 1))));
    case ModelType.LG
        params = {};
        train_labels(train_labels == -1) = 0;
        test_labels(test_labels == -1) = 0;
    otherwise
        error('Unsupported ModelType');
end

% Create the model object
model = ModelFactory(modelType, featureType, preprocessingType, params);

% Train the model
model = model.train(train_images, train_labels);

% Test the model
predictions = model.test(test_images);

% Evaluate the model
model.evaluate(predictions, test_labels, test_images);

% Save the trained model
savePath = sprintf('saved-models/%s/%s_%s_Model.mat', ...
    char(modelType), char(featureType), char(preprocessingType));
save(savePath, 'model');
