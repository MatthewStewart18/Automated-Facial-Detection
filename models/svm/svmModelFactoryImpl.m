clear all
close all

% Add relevant files to current WD
addpath ../
addpath SVM-KM
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils
addpath ../../feature-extraction-utils/feature-extractors
addpath ../../preprocessing-utils

% Load training and test data
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset');
[test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset');

% Define model-specific functions
trainingFunc = @SVMtraining;
predictionFunc = @extractPredictionsSVM;
params = struct('kerneloption', 3.85, 'kernel', 'polyhomog');

% Create the model object
svmModel = ModelFactory(trainingFunc, predictionFunc, params);

% Add preprocessing steps
svmModel = svmModel.addPreprocessingStep(@histEq);

% Add preprocessing steps
svmModel = svmModel.addFeatureExtractionStep(@extractEdges);
svmModel = svmModel.addFeatureExtractionStep(@extractPcaDim, 150);

% Train the model
svmModel = svmModel.train(train_images, train_labels);

% Test the model
predictions = svmModel.test(test_images);

% Evaluate the model
svmModel.evaluate(predictions, test_labels, test_images);

% save current model
save('saved-models/modelSVMEdgesPca.mat', 'svmModel');
