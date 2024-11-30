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

% Load training data without augmentation
[Xtrain, Ytrain] = loadFaceImages('../../images/face_train.cdataset', -1);
[Xtest, Ytest] = loadFaceImages('../../images/face_test.cdataset', -1);

% Concatenate images vertically (rows correspond to images)
images = [Xtrain; Xtest];
labels = [Ytrain; Ytest];
% Augment image set
[images, labels] = augmentData(images, labels, [27, 18]);

% Set current model and feature configurations
modelType = ModelType.RF;
featureType = FeatureType.HOG;
preprocessingType = PreprocessingType.HistEq;

% Define model parameters
switch modelType
    case ModelType.SVM
        params = struct('kerneloption', 2, 'kernel', 'polyhomog');
    case ModelType.KNN
        params = struct('K', round(sqrt(size(train_labels, 1))));
    case ModelType.LG
        params = {};
        train_labels(train_labels == -1) = 0;
        test_labels(test_labels == -1) = 0;
    case ModelType.RF
        numTrees = 275;
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
model = model.train(images, labels);

% Save the trained model
savePath = sprintf('saved-final-models/%s/%s_%s_Model.mat', ...
    char(modelType), char(featureType), char(preprocessingType));

save(savePath, 'model');