% Read the images
image1 = imread('data/edges3D.png');
image2 = imread('data/pca3d.png');

% Create a figure
figure;

% Display the first image
subplot(1, 2, 1); % 1 row, 2 columns, position 1
imshow(image1);

% Display the second image
subplot(1, 2, 2); % 1 row, 2 columns, position 2
imshow(image2);
