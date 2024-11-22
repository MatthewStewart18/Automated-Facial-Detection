function [gaborFeatures] = extractGabor(images, image_size)
    % Check if image_size is provided, default to [27, 18]
    if nargin < 2
        image_size = [27, 18];
    end

    % Load precomputed Gabor filters
    load gabor;
    
    % Number of images
    num_images = size(images, 1);

    % Initialize feature matrix
    % Assume each image produces a fixed-length feature vector (e.g., 19440)
    temp_image = reshape(images(1, :), image_size); % Use one image to infer size
    temp_features = extractGaborSingle(temp_image, G);
    gaborFeatures = zeros(num_images, numel(temp_features));
    
    % Process each image
    for i = 1:num_images
        img = reshape(images(i, :), image_size); % Reshape flattened image to original size
        gaborFeatures(i, :) = extractGaborSingle(img, G); % Extract features
    end
end

function [vector] = extractGaborSingle(image, G)
    % Perform preprocessing on the image
    image = adapthisteq(image, 'Numtiles', [8, 3]);

    % Initialize feature storage
    Features135x144 = cell(5, 8);
    
    % Compute Gabor features
    for s = 1:5
        for j = 1:8
            % Apply Gabor filter
            Features135x144{s, j} = abs(ifft2(G{s, j} .* fft2(double(image), 32, 32), 27, 18));
        end
    end
    
    % Combine features into a single vector
    Features45x48 = cell2mat(Features135x144);
    vector = reshape(Features45x48, [1, numel(Features45x48)]);
end
