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

% Load training data without augmentation
[Xtrain, Ytrain] = loadFaceImages('../../images/face_train.cdataset', -1);
[Xtest, Ytest] = loadFaceImages('../../images/face_test.cdataset', -1);

% Concatenate images vertically (rows correspond to images)
images = [Xtrain; Xtest];
labels = [Ytrain; Ytest];

figure;
overlapping_idx = {3, 4, 101, 102, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 98, 99};
for i=1:size(overlapping_idx, 2)
    subplot(4, 8, i);
    imshow(uint8(reshape(images(overlapping_idx{i}, :), image_size))); 
    title(sprintf('Image %d', overlapping_idx{i}));
end
sgtitle("Similar faces in Training set")
