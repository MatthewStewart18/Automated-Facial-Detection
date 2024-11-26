clear all
close all

% Add relevant files to WD
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils
addpath ../../feature-extraction-utils/feature-extractors

% Load training and test data without augmentation
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset', -1);

% Feature extraction function
featureFunc = @extractEdgesContinuous;
featureFunc2 = @extractPca;
featureArgs = {};

% Extract features
train_images = preProcess(train_images, @meanFilter, 2);
train_images = preProcess(train_images, @histEq);
features = featureExtraction(train_images, featureFunc);
% features = featureExtraction(features, featureFunc2);

% Find a face and a non-face sample
face_idx = find(train_labels == 1, 1); % Index of a face image
non_face_idx = find(train_labels == -1, 3); % Index of a non-face image
non_face_idx = non_face_idx(2);
image_size = [27, 18];

% Display original and processed images for a face
figure;
subplot(2, 2, 1);
imshow(uint8(reshape(train_images(face_idx, :), image_size))); 
title('Original Face Image');
subplot(2, 2, 2);
imshow(reshape(features(face_idx, :), image_size));
title('Feature Extracted Face');

% Display original and processed images for a non-face
subplot(2, 2, 3);
imshow(uint8(reshape(train_images(non_face_idx, :), image_size)));
title('Original Non-Face Image');
subplot(2, 2, 4);
imshow(reshape(features(non_face_idx, :), image_size));
title('Feature Extracted Non-Face');
