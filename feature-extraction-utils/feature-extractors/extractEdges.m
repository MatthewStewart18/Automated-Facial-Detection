function [edges] = extractEdges(images, image_size)
    % Check if window_size is provided, if not, default to [27, 18]
    if nargin < 2
        image_size = [27, 18];
    end
    
    num_images = size(images, 1);
    edges = zeros(num_images, prod(image_size)); 

    for i = 1:num_images
        img = reshape(images(i,:), image_size);
        % Extract edges using Canny
        current_edge = edge(double(img), 'Canny');
        current_edge = current_edge(:)';
        edges(i, :) = double(current_edge);
    end
end


