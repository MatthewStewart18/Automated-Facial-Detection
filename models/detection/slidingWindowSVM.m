clear all
close all
addpath ../svm
addpath ../svm/SVM-KM
addpath ../svm/saved-models
addpath ../../detection-images
addpath ../../utils
addpath ../../feature-extraction-utils/feature-extractors

% Load the trained SVM model
load('../svm/saved-models/modelSVMEdges.mat', 'modelSVM');

% Parameters for sliding window
window_size = [27, 18];
step_size = 1;
scales = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7];
confidence_threshold = 0.7; % Minimum confidence for detections

% Loop over each test image
for img_num = 1:1
    img_name = sprintf('im%d.jpg', img_num);
    img = imread(img_name);
    detections = [];
    
    for scale = scales
        resized_img = imresize(img, scale);
        [resized_height, resized_width] = size(resized_img);
        scaled_window_size = round(window_size * scale);
        
        for y = 1:step_size:resized_height - window_size(1)
            for x = 1:step_size:resized_width - window_size(2)
                window = resized_img(y:y+window_size(1)-1, x:x+window_size(2)-1);
                window_vector = reshape(window, 1, []);
                window_edges = extractEdges(window_vector);
                
                % Get both label and confidence
                [is_face, confidence] = extractPredictionsSVM(window_edges, modelSVM);
                
                if is_face && confidence > confidence_threshold
                    bbox = [x / scale, y / scale, window_size(2) / scale, window_size(1) / scale, confidence];
                    detections = [detections; bbox];
                end
            end
        end
    end
    
    % Apply Non-Maximum Suppression
    detections = simpleNMS(detections, 0.75);
    ShowDetectionResult(img, detections);
end
