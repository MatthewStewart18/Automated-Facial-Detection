function [output_images] = meanFilterUnshaped(images, N)
    % Initialize output_images as a cell array of the same size as input
    output_images = cell(size(images));
    
    % Create the averaging filter
    B = ones(N, N) / (N^2); % Mean filter kernel

    % Process each image in the list
    for i = 1:numel(images)
        % Extract the i-th image
        img = images{i};
        
        % Apply the mean filter
        filtered_img = filter2(B, double(img), 'same');
        
        % Store the filtered image in the output list
        output_images{i} = uint8(filtered_img); % Convert back to uint8
    end
end
