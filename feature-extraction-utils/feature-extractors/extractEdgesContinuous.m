function [edges] = extractEdgesContinuous(images, image_size)
    % Check if window_size is provided, if not, default to [27, 18]
    if nargin < 2
        image_size = [27, 18];
    end
    
    num_images = size(images, 1);
    edges = zeros(num_images, prod(image_size)); 

    for i = 1:num_images
        img = reshape(images(i,:), image_size);
        % Compute gradients in x and y directions using imgradientxy
        [gradient_x, gradient_y] = imgradientxy(img);
     
        % Calculate the gradient magnitude (edge strength)
        gradient_magnitude = sqrt(gradient_x.^2 + gradient_y.^2);
        
        % Flatten the gradient magnitude matrix and store it
        edges(i, :) = gradient_magnitude(:)';
    end
    edges = normalize(edges, 'zscore');
end


