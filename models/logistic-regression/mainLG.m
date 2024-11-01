clear all
close all
addpath ../../images

% Add the folder containing loadFaceImages to the MATLAB path
addpath('/Users/samuelagnew/Documents/year3Term1/csc3067/CSC3067-2425-G3/utils');


% Load training and testing datasets
[trainingImages, trainingLabels] = loadFaceImages("../../images/face_train.cdataset", 1);
[testingImages, testingLabels] = loadFaceImages("../../images/face_test.cdataset", 1);
numTrainingImages = size(trainingImages, 1);
numTestImages = size(testingImages, 1);

featureMatrix = zeros(numTrainingImages, 3888);


% Add the folder containing loadFaceImages to the MATLAB path
addpath('/Users/samuelagnew/Documents/year3Term1/csc3067/CSC3067-2425-G3/feature-extraction-utils/gabor');

imageList = reshape2dImage(trainingImages);
for imageNum = 1:length(imageList)
    image = cell2mat(imageList(imageNum));
    featureMatrix(imageNum, :) = gabor_feature_vector_subset(image);
end
[~, ~, featureMatrix] = pca(featureMatrix, 2);