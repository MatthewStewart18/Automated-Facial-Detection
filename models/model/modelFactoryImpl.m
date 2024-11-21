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
featureType = FeatureType.RawPix;

% Define model parameters
switch modelType
    case ModelType.SVM
        params = struct('kerneloption', 8, 'kernel', 'polyhomog');
    case ModelType.KNN
        params = struct('K', 5);
    otherwise
        error('Unsupported ModelType');
end

% Create the model object
model = ModelFactory(modelType, featureType, params);

% Add preprocessing steps
model = model.addPreprocessingStep(@histEq);

% Train the model
model = model.train(train_images, train_labels);

% Test the model
predictions = model.test(test_images);

% Evaluate the model
model.evaluate(predictions, test_labels, test_images);

% Save the trained model
savePath = sprintf('saved-models/%s_%s_Model.mat', char(modelType), char(featureType));
save(savePath, 'model');
