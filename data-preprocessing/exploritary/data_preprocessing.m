clear all
close all

addpath ../../preprocessing-utils
addpath ../../preprocessing-utils/brightness-enhancement
addpath ../../preprocessing-utils/hist-eq
addpath ../../preprocessing-utils/linear-stretching
addpath ../../preprocessing-utils/power-law
addpath ../../images
addpath ../../utils

% Load training and test data
[preprocessingimages, preprocessinglabels] = loadFaceImages('../../images/data-preprocessing-test.cdataset');
preprocessingimages = reshape2dImage(preprocessingimages);

% Separate images by label
indicesLabel1 = find(preprocessinglabels == 1);       % Face images
indicesLabelMinus1 = find(preprocessinglabels == -1); % Non-face images

% -------------------- Face Images (4x4 grid) -------------------- %
figure;
sgtitle('Face Images After Augmentation');

% Display 10 face images in a 4x4 grid (with last two as blanks)
for i = 1:10
    subplot(4, 4, i);  % 4x4 grid
    imshow(preprocessingimages{indicesLabel1(i)}, []);
    title(sprintf('Face %d', i));
end

% -------------------- Non-Face Images (2x2 grid) -------------------- %
figure;
sgtitle('Non-Face Images After Augmentation');

% Display up to 4 non-face images in a 2x2 grid
for i = 1:min(4, length(indicesLabelMinus1))
    subplot(2, 2, i);  % 2x2 grid
    imshow(preprocessingimages{indicesLabelMinus1(i)}, []);
    title(sprintf('Non-Face %d', i));
end



% % Load training and test data
% [train_images, train_labels] = loadFaceImages('../../images/face_train.cdataset');
% [test_images, test_labels] = loadFaceImages('../../images/face_test.cdataset');
% 
% 
% imagelist = reshape2dImage(train_images);
% % Define the sampling fraction (e.g., 20% of each class)
% sampleFraction = 0.01;
% 
% % Unique labels and their indices
% [uniqueLabels, ~, labelGroups] = unique(train_labels);
% 
% % Preallocate sampled indices
% sampledIndices = [];
% 
% % Randomly sample indices for each label group
% for label = uniqueLabels'
%     % Find indices for the current label
%     labelIndices = find(train_labels == label);
% 
%     % Calculate the number of samples to take
%     numSamples = round(length(labelIndices) * sampleFraction);
% 
%     % Randomly select indices
%     sampledIndices = [sampledIndices; labelIndices(randperm(length(labelIndices), numSamples))];
% end
% 
% % Extract the stratified sample
% sampledImages = imagelist(sampledIndices);
% sampledLabels = train_labels(sampledIndices);
% 
% % Display the first 5 images and their labels as an example
% figure;
%  numRows = 4; % Number of rows
%  numCols = 6; % Number of columns
% 
% for imgIndex = 1:min(24, length(sampledImages))
%     % Convert image to uint8 if needed
%     img = uint8(sampledImages{imgIndex});
% 
%     % Create a subplot with 4 rows and 5 columns
%     subplot(numRows, numCols, imgIndex);
% 
%     % Display the image
%     imshow(img);
% 
%     % Add the corresponding label as the title
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% end
% sgtitle('Comparison of Original and Brightness Enhanced Images');
% hold off;
% 
% %%apply brightness-enhancement to sampled images and show
% 
% % Define fixed scales
% xRange = [0 255]; % Fixed x-axis range for pixel intensities
% yRange = [0 20];
% 
% % Preallocate cell array
% sampledImagesBrightness = cell(size(sampledImages));
% 
% for img = 1:length(sampledImages)
%     % Extract the image from the cell and enhance brightness
%     enhancedImg = enhanceBrightness(uint8(sampledImages{img}), 50); 
%     % If you want to store the enhanced images, save them in a new cell array
%     sampledImagesBrightness{img} = enhancedImg;
% end
% 
% % Number of images to compare
% numToCompare = 5; % Adjust this to the number of images to compare
% numCols = numToCompare; % Each image gets its own column
% 
% % Create a new figure
% figure;
% 
% % Loop to display original images (top 2 rows)
% for imgIndex = 1:numToCompare
%     % Extract the original image
%     originalImg = uint8(sampledImages{imgIndex});
% 
%     % Display the original image in the first row
%     subplot(4, numCols, imgIndex); % First row
%     imshow(originalImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the original image in the second row
%     subplot(4, numCols, numCols + imgIndex); % Second row
%     [counts, binLocations] = imhist(originalImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% 
% % Loop to display enhanced images and their histograms
% for imgIndex = 1:numToCompare
%     % Extract the enhanced image
%     enhancedImg = uint8(sampledImagesBrightness{imgIndex});
% 
%     % Display the enhanced image in the third row
%     subplot(4, numCols, 2*numCols + imgIndex); % Third row
%     imshow(enhancedImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the enhanced image in the fourth row
%     subplot(4, numCols, 3*numCols + imgIndex); % Fourth row
%     [counts, binLocations] = imhist(enhancedImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% sgtitle('Comparison of Original and Brightness Enhanced Images');
% hold off;
% 
% %%hist-eq
% 
% 
% % Preallocate cell array
% sampledImagesHistEq = cell(size(sampledImages));
% 
% for img = 1:length(sampledImages)
%     % Extract the image from the cell and enhance brightness
%     enhancedImg = enhanceContrastHE(uint8(sampledImages{img}));
%     % If you want to store the enhanced images, save them in a new cell array
%     sampledImagesHistEq{img} = enhancedImg;
% end
% 
% 
% % Create a new figure
% figure;
% 
% % Loop to display original images (top 2 rows)
% for imgIndex = 1:numToCompare
%     % Extract the original image
%     originalImg = uint8(sampledImages{imgIndex});
% 
%     % Display the original image in the first row
%     subplot(4, numCols, imgIndex); % First row
%     imshow(originalImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the original image in the second row
%     subplot(4, numCols, numCols + imgIndex); % Second row
%     [counts, binLocations] = imhist(originalImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% 
% % Loop to display enhanced images and their histograms
% for imgIndex = 1:numToCompare
%     % Extract the enhanced image
%     enhancedImg = uint8(sampledImagesHistEq{imgIndex});
% 
%     % Display the enhanced image in the third row
%     subplot(4, numCols, 2*numCols + imgIndex); % Third row
%     imshow(enhancedImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the enhanced image in the fourth row
%     subplot(4, numCols, 3*numCols + imgIndex); % Fourth row
%     [counts, binLocations] = imhist(enhancedImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% sgtitle('Comparison of Original and Histogram Equalisation Enhanced Images');
% hold off;
% 
% 
% %%linear-stretching
% 
% % Preallocate cell array
% sampledImageslinearStretch = cell(size(sampledImages));
% 
% for img = 1:length(sampledImages)
%     % Extract the image from the cell and enhance brightness 
%     enhancedImg = enhanceContrastALS(uint8(sampledImages{img}));
% 
%    % If you want to store the enhanced images, save them in a new cell array
%     sampledImageslinearStretch{img} = enhancedImg;
% end
% 
% % Create a new figure
% figure;
% 
% % Loop to display original images (top 2 rows)
% for imgIndex = 1:numToCompare
%     % Extract the original image
%     originalImg = uint8(sampledImages{imgIndex});
% 
%     % Display the original image in the first row
%     subplot(4, numCols, imgIndex); % First row
%     imshow(originalImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the original image in the second row
%     subplot(4, numCols, numCols + imgIndex); % Second row
%     [counts, binLocations] = imhist(originalImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% 
% % Loop to display enhanced images and their histograms
% for imgIndex = 1:numToCompare
%     % Extract the enhanced image
%     enhancedImg = uint8(sampledImageslinearStretch{imgIndex});
% 
%     % Display the enhanced image in the third row
%     subplot(4, numCols, 2*numCols + imgIndex); % Third row
%     imshow(enhancedImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the enhanced image in the fourth row
%     subplot(4, numCols, 3*numCols + imgIndex); % Fourth row
%     [counts, binLocations] = imhist(enhancedImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% sgtitle('Comparison of Original and Linear-Stretched Enhanced Images');
% hold off;
% %%power-law >1
% 
% % Preallocate cell array
% sampledImagesPowerLaw15 = cell(size(sampledImages));
% 
% for img = 1:length(sampledImages)
%     % Extract the image from the cell and enhance brightness
%     enhancedImg = enhanceContrastPL(uint8(sampledImages{img}), 1.5);
% 
%    % If you want to store the enhanced images, save them in a new cell array
%     sampledImagesPowerLaw15{img} = enhancedImg;
% end
% 
% % Create a new figure
% figure;
% 
% % Loop to display original images (top 2 rows)
% for imgIndex = 1:numToCompare
%     % Extract the original image
%     originalImg = uint8(sampledImages{imgIndex});
% 
%     % Display the original image in the first row
%     subplot(4, numCols, imgIndex); % First row
%     imshow(originalImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the original image in the second row
%     subplot(4, numCols, numCols + imgIndex); % Second row
%     [counts, binLocations] = imhist(originalImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% 
% % Loop to display enhanced images and their histograms
% for imgIndex = 1:numToCompare
%     % Extract the enhanced image
%     enhancedImg = uint8(sampledImagesPowerLaw15{imgIndex});
% 
%     % Display the enhanced image in the third row
%     subplot(4, numCols, 2*numCols + imgIndex); % Third row
%     imshow(enhancedImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the enhanced image in the fourth row
%     subplot(4, numCols, 3*numCols + imgIndex); % Fourth row
%     [counts, binLocations] = imhist(enhancedImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% sgtitle('Comparison of Original and Power-Law > 1 (1.5) Enhanced Images');
% hold off;
% %%power-law <1
% 
% % Preallocate cell array
% sampledImagesPowerLaw05 = cell(size(sampledImages));
% 
% for img = 1:length(sampledImages)
%     % Extract the image from the cell and enhance brightness
%     enhancedImg = enhanceContrastPL(uint8(sampledImages{img}), 0.5);
% 
%    % If you want to store the enhanced images, save them in a new cell array
%     sampledImagesPowerLaw05{img} = enhancedImg;
% end
% 
% % Create a new figure
% figure;
% 
% for imgIndex = 1:numToCompare
%     % Extract the original image
%     originalImg = uint8(sampledImages{imgIndex});
% 
%     % Display the original image in the first row
%     subplot(4, numCols, imgIndex); % First row
%     imshow(originalImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the original image in the second row
%     subplot(4, numCols, numCols + imgIndex); % Second row
%     [counts, binLocations] = imhist(originalImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% 
% % Loop to display enhanced images and their histograms
% for imgIndex = 1:numToCompare
%     % Extract the enhanced image
%     enhancedImg = uint8(sampledImagesPowerLaw05{imgIndex});
% 
%     % Display the enhanced image in the third row
%     subplot(4, numCols, 2*numCols + imgIndex); % Third row
%     imshow(enhancedImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the enhanced image in the fourth row
%     subplot(4, numCols, 3*numCols + imgIndex); % Fourth row
%     [counts, binLocations] = imhist(enhancedImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% sgtitle('Comparison of Original and Power-Law < 1 (0.5) Enhanced Images');
% hold off;
% 
% % Preallocate cell array
% sampledImagesMeanFilter = cell(size(sampledImages));
% sampledImagesMeanFilter = meanFilterUnshaped(sampledImages, 2);
% 
% % Create a new figure
% figure;
% 
% % Loop to display original images (top 2 rows)
% for imgIndex = 1:numToCompare
%     % Extract the original image
%     originalImg = uint8(sampledImages{imgIndex});
% 
%     % Display the original image in the first row
%     subplot(4, numCols, imgIndex); % First row
%     imshow(originalImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the original image in the second row
%     subplot(4, numCols, numCols + imgIndex); % Second row
%     [counts, binLocations] = imhist(originalImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% 
% % Loop to display enhanced images and their histograms
% for imgIndex = 1:numToCompare
%     % Extract the enhanced image
%     enhancedImg = uint8(sampledImagesMeanFilter{imgIndex});
% 
%     % Display the enhanced image in the third row
%     subplot(4, numCols, 2*numCols + imgIndex); % Third row
%     imshow(enhancedImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the enhanced image in the fourth row
%     subplot(4, numCols, 3*numCols + imgIndex); % Fourth row
%     [counts, binLocations] = imhist(enhancedImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% sgtitle('Comparison of Original and Mean Filter Enhanced Images');
% hold off;
% 
% %%medianfilter
% % Preallocate cell array
% sampledImagesMedianFilter = cell(size(sampledImages));
% sampledImagesMedianFilter = medianFilterUnshaped(sampledImages, 2);
% 
% 
% % Create a new figure
% figure;
% 
% % Loop to display original images and their histograms
% for imgIndex = 1:numToCompare
%     % Extract the original image
%     originalImg = uint8(sampledImages{imgIndex});
% 
%     % Display the original image in the first row
%     subplot(4, numCols, imgIndex); % First row
%     imshow(originalImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the original image in the second row
%     subplot(4, numCols, numCols + imgIndex); % Second row
%     [counts, binLocations] = imhist(originalImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% 
% % Loop to display enhanced images and their histograms
% for imgIndex = 1:numToCompare
%     % Extract the enhanced image
%     enhancedImg = uint8(sampledImagesMedianFilter{imgIndex});
% 
%     % Display the enhanced image in the third row
%     subplot(4, numCols, 2*numCols + imgIndex); % Third row
%     imshow(enhancedImg);
%     title(['Label: ', num2str(sampledLabels(imgIndex))]);
% 
%     % Display the histogram of the enhanced image in the fourth row
%     subplot(4, numCols, 3*numCols + imgIndex); % Fourth row
%     [counts, binLocations] = imhist(enhancedImg); % Compute histogram
%     bar(binLocations, counts, 'FaceColor', 'w', 'EdgeColor', 'k'); % White bins, black edges
%     xlim(xRange); % Fixed x-axis
%     ylim(yRange); % Fixed y-axis
% end
% 
% % Add a main title to the figure
% sgtitle('Comparison of Original and  Median Filter Enhanced Images with Histograms');
% hold off;
