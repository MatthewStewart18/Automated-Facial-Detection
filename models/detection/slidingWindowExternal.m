clear all
close all
addpath ../svm
addpath ../svm/SVM-KM
addpath ../../detection-images
addpath ../../utils

% Load the trained SVM model
load('modelSVM.mat', 'modelSVM');

% Parameters for sliding window
window_size = [27, 18]; % Match training window size
step_size = 20; % Move the window 5 pixels each time (adjust for balance of speed vs. accuracy)
scales = [1.0]; % Example scales to detect faces of different sizes

% Loop over each test image (e.g., im1.jpg to im4.jpg)
for img_num = 1:1
    % Load the image
    img_name = sprintf('im%d.jpg', img_num);
    img = imread(img_name);
    
    [bbox,score] = detect(modelSVM, img, 'SelectStrongest', false);
    
    % Apply Non-Maximum Suppression to reduce overlapping detections
    [final_bboxes, ~] = selectStrongestBbox(bbox,score,'OverlapThreshold',0.3);
    
    % Display detections on the original image
    figure;
    imshow(img);
    hold on;
    for i = 1:size(final_bboxes, 1)
        rectangle('Position', final_bboxes(i, :), 'EdgeColor', 'r', 'LineWidth', 2);
    end
    hold off;
    title(sprintf('Detected Faces in %s', img_name));
end
