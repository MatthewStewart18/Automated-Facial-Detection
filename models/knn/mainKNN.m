clear all
close all
addpath ../../images

% each row in this matrix is a flattened version of an image in the sample
trainingImages = loadFaceImages("../../images/face_train.cdataset", 1);