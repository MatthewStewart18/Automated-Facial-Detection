clear all
close all

% Add relevant files to WD
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils
addpath ../../feature-extraction-utils/feature-extractors

% Load training and test data without augmentation
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset', -1);

% Image size
image_size = [27, 18];

% Find 3 faces and 3 non-faces
face_idx = find(train_labels == 1, 20); % Indices of 3 face images
non_face_idx = find(train_labels == -1, 20); % Indices of 3 non-face images

% Display original images for 3 faces
figure;
subplot(2, 3, 1);
imshow(uint8(reshape(train_images(face_idx(10), :), image_size))); 
title('Original Face 1');
subplot(2, 3, 2);
imshow(uint8(reshape(train_images(face_idx(1), :), image_size))); 
title('Original Face 2');
subplot(2, 3, 3);
imshow(uint8(reshape(train_images(face_idx(15), :), image_size))); 
title('Original Face 3');

% Display original images for 3 non-faces
subplot(2, 3, 4);
imshow(uint8(reshape(train_images(non_face_idx(3), :), image_size)));
title('Original Non-Face 1');
subplot(2, 3, 5);
imshow(uint8(reshape(train_images(non_face_idx(19), :), image_size)));
title('Original Non-Face 2');
subplot(2, 3, 6);
imshow(uint8(reshape(train_images(non_face_idx(18), :), image_size)));
title('Original Non-Face 3');
