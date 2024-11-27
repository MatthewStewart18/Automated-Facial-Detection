function [output_images] = medianFilterUnshaped(images, N)
    % Initialize output_images as a cell array of the same size as input
    output_images = cell(size(images));
    
    % Process each image in the list
    for i = 1:numel(images)
        % Extract the i-th image
        img = images{i};
        
        % Apply the median filter
        filtered_img = medfilt2(img, [N, N]);
        
        % Store the filtered image in the output list
        output_images{i} = uint8(filtered_img); % Convert to uint8 if needed
    end
end
