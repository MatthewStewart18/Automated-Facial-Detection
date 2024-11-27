clear all 
close all 

% Add relevant files to WD
addpath ../../images
addpath ../../utils
addpath ../../preprocessing-utils
addpath ../../feature-extraction-utils
addpath ../../feature-extraction-utils/feature-extractors

% Load training and test data without augmentation
[train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset');

h=27;
w=18;

% draw the first 9 images 
figure
for i=1:9
    Im = reshape(train_images(i,:),h,w);
    subplot(3,3,i), imagesc(Im), colormap gray; axis off;
end

% Apply feature extraction
% train_images = histEq(train_images);
% train_images = extractHog(train_images);

% Apply PCA
[eigenVectors, eigenvalues, meanX, Xpca] = extractPca(train_images, 15);

%% show 0th through 15th principal eigenvectors 
eig0 = reshape(meanX, [h,w]); 
figure,subplot(4,4,1) 
imagesc(eig0) 
colormap gray 
for i = 1:15 
    subplot(4,4,i+1) 
    imagesc(reshape(eigenVectors(:,i),h,w)) 
end

