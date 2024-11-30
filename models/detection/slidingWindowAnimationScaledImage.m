clear all;
close all;

% Add required directories to path
addpath ../svm
addpath ../svm/SVM-KM
addpath ../knn
addpath ../model
addpath ../../images
addpath ../../utils
addpath ../../detection-images
addpath ../../preprocessing-utils
addpath ../../feature-extraction-utils
addpath ../../feature-extraction-utils/feature-extractors

% Set current Model
modelType = ModelType.SVM;
featureType = FeatureType.EdgesPCA;
preprocessingType = PreprocessingType.None;

% Load Model
path = sprintf('../model/saved-models/%s/%s_%s_Model.mat', ...
    char(modelType), char(featureType), char(preprocessingType));
load(path, 'model');


% Load the test image
testImage = imread('../../detection-images/im1.jpg');
originalSize = size(testImage);

% sliding window parameters
windowSize = [27, 18];
stepSize = [5, 5];
scales = [0.9, 1, 1.1];

[rows, cols] = size(testImage);


for scale = scales
    tic
    fprintf('Scale: %.2f\n', scale)
    scaledImage = imresize(testImage, scale, 'bicubic');
    [rows, cols] = size(scaledImage);
    fprintf('Image dimensions: %dx%d\n', rows, cols);

    % Calculate output grid dimensions
    outRows = ceil((rows - windowSize(1) + 1) / stepSize(1));
    outCols = ceil((cols - windowSize(2) + 1) / stepSize(2));

    figure('Name', sprintf('Scale %.2f - Sliding Window Animation', scale));
    imshow(scaledImage, []);
    hold on;

    % Iterate over the sliding window positions
    for outRow = 1:outRows
        for outCol = 1:outCols
            % Calculate the current window position
            currentRow = (outRow - 1) * stepSize(1) + 1;
            currentCol = (outCol - 1) * stepSize(2) + 1;
    
            % Check if the window exceeds image bounds
            if currentRow + windowSize(1) - 1 > rows || currentCol + windowSize(2) - 1 > cols
                continue;
            end
    
            % Draw the current bounding box
            rectangle('Position', [currentCol, currentRow, windowSize(2), windowSize(1)], ...
                      'EdgeColor', 'r', 'LineWidth', 1);
    
            % Pause for animation effect
            pause(0.05);
    
            % Remove the rectangle for the next iteration
            delete(findall(gca, 'Type', 'rectangle'));
        end
    end
    toc
end
