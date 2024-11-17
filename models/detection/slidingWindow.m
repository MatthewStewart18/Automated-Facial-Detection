clear all
close all
addpath ../knn
addpath ../../detection-images
addpath ../../utils

% Load the trained KNN model
load('modelKNN.mat', 'modelKNN', 'K');

% Parameters for sliding window
window_size = [27, 18]; % Match training window size
step_size = 10; % Move the window 5 pixels each time (adjust for balance of speed vs. accuracy)
scales = [1.0, 0.8, 0.6, 0.4]; % Example scales to detect faces of different sizes

% Loop over each test image (e.g., im1.jpg to im4.jpg)
for img_num = 1:4
    % Load the image
    img_name = sprintf('im%d.jpg', img_num);
    img = imread(img_name);
    
    % Initialize list for detections
    detections = [];
    scores = [];
    
    % Loop over each scale
    for scale = scales
        % Resize image for current scale
        resized_img = imresize(img, scale);
        [resized_height, resized_width] = size(resized_img);
        
        % Loop over the image with the sliding window
        for y = 1:step_size:resized_height - window_size(1)
            for x = 1:step_size:resized_width - window_size(2)
                % Extract the window region
                window = resized_img(y:y+window_size(1)-1, x:x+window_size(2)-1);
                window_vector = reshape(window, 1, []);
                
                % Extract features from the window (e.g., edges)
                window_edges = extractEdges(window_vector);
                
                % Classify the window using the KNN model
                is_face = KNNTesting(window_vector, modelKNN, K);
                
                % If classified as a face, save the bounding box information
                if is_face
                    % Scale the bounding box coordinates back to the original image size
                    bbox = [x / scale, y / scale, window_size(2) / scale, window_size(1) / scale];
                    detections = [detections; bbox];
                    scores = [scores; 1];
                end
            end
        end
    end
    
    % Apply Non-Maximum Suppression to reduce overlapping detections
    final_bboxes = selectStrongestBbox(detections, scores, 'RatioType', 'Union');
    
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
