function [lbpFeatures] = extractLBP(images, image_size)

    if nargin < 2
        image_size = [27, 18];
    end
    
    num_images = size(images, 1);
    
    % Does not work without normalizing
    % Its comparing neighbouring pixels so im assuming large differences
    % in pixel values is the issue. Patterns can then represent relative
    % differences when features are compressed to range (0-1)

    if max(images(:)) > 1
        images = double(images)/255;
    end
    
    % init feature matrix
    temp_image = reshape(images(1, :), image_size);
    temp_features = extractLBPSingle(temp_image);
    lbpFeatures = zeros(num_images, numel(temp_features));
    
    for i = 1:num_images
        img = reshape(images(i, :), image_size);
        lbpFeatures(i, :) = extractLBPSingle(img);
    end
end

function [vector] = extractLBPSingle(image)

    image = adapthisteq(image, 'NumTiles', [8 8]);
    
    lbpFeatures = extractLBPFeatures(image, ...
        'CellSize', [6 6], ...
        'Radius', 1, ...
        'NumNeighbors', 8);

    vector = reshape(lbpFeatures, 1, []);
end