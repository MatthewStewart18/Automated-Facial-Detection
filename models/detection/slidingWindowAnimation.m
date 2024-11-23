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

% sliding window parameters
windowSize = [27, 18];
stepSize = [5, 5];
scales = [1, 0.7];
[rows, cols] = size(testImage);

for scale = scales
    tic
    scaled_window_size = round(windowSize * scale);

    % Calculate output grid dimensions
    outRows = ceil((rows - scaled_window_size(1) + 1) / stepSize(1));
    outCols = ceil((cols - scaled_window_size(2) + 1) / stepSize(2));

    figure('Name', 'Sliding Window Animation');
    imshow(testImage, []);
    hold on;
    % Iterate over the sliding window positions
    for outRow = 1:outRows
        for outCol = 1:outCols
            % Calculate the current window position
            currentRow = (outRow - 1) * stepSize(1) + 1;
            currentCol = (outCol - 1) * stepSize(2) + 1;
    
            % Check if the window exceeds image bounds
            if currentRow + scaled_window_size(1) - 1 > rows || currentCol + scaled_window_size(2) - 1 > cols
                continue;
            end
    
            % Draw the current bounding box
            rectangle('Position', [currentCol, currentRow, scaled_window_size(2), scaled_window_size(1)], ...
                      'EdgeColor', 'r', 'LineWidth', 1);
    
            % Pause for animation effect
            pause(0.05);
    
            % Remove the rectangle for the next iteration
            delete(findall(gca, 'Type', 'rectangle'));
        end
    end
    toc
end
