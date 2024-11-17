function Objects = KNNObjectDetection(image, modelKNN, K, Options)
% Objects = KNNObjectDetection(image, modelKNN, K, Options)
%
% Inputs:
%   image: The grayscale image to scan for objects
%   modelKNN: Trained KNN model for classification
%   K: Number of nearest neighbors for KNN
%   Options: Struct with options (e.g., scale update, verbose mode)
%
% Output:
%   Objects: Nx5 array with detected object bounding boxes and confidence scores
%            in the format [x y width height confidence]

% Array to store [x y width height confidence] of detected objects
Objects = zeros(100, 5); n = 0;

% Calculate the coarsest scale factor
ScaleWidth = size(image, 2) / Options.windowSize(1);
ScaleHeight = size(image, 1) / Options.windowSize(2);
StartScale = min(ScaleWidth, ScaleHeight);

% Calculate maximum number of scale iterations
itt = ceil(log(1 / StartScale) / log(Options.ScaleUpdate));

% Loop over all image scales
for i = 1:itt
    % Current scale
    Scale = StartScale * Options.ScaleUpdate^(i - 1);
    
    % Display current scale and number of objects detected
    if Options.Verbose
        disp(['Scale : ' num2str(Scale) ' objects detected : ' num2str(n)]);
    end
    
    % Set window size for current scale
    w = floor(Options.windowSize(1) * Scale);
    h = floor(Options.windowSize(2) * Scale);

    % Spacing between search coordinates
    step = floor(max(Scale, 2));

    % Generate grid of search coordinates for current scale
    [x, y] = ndgrid(0:step:(size(image, 2) - w - 1), 0:step:(size(image, 1) - h - 1));
    x = x(:); y = y(:);
    
    % Loop over each coordinate (x, y) to perform classification
    for k = 1:length(x)
        % Extract window region at the current location and scale
        window = image(y(k) + (1:h), x(k) + (1:w));
        
        % Resize window to match training data size if necessary
        window = imresize(window, Options.windowSize);
        
        % Extract features from the window
        features = extractEdges(window); % Replace with your feature extraction
        
        % Classify the window using KNN
        label = KNNTesting(features(:)', modelKNN, K); % Ensure features are a row vector
        
        % If classified as a face (assuming label "1" indicates a face)
        if label == 1
            % Set confidence score as 1 (placeholder for KNN)
            confidence = 1; 
            
            % Save detected bounding box with confidence score
            n = n + 1;
            Objects(n, :) = [x(k), y(k), w, h, confidence];
        end
    end
end

% Crop the initial array with detected objects
Objects = Objects(1:n, :);

end
