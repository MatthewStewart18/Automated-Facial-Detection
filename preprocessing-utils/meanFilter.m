function [output_images] = meanFilter(images, N)
    output_images = zeros(size(images, 1), 27*18);

    if nargin < 2
        N = 2;
    end

    B = ones(N, N) / (N^2); %averaging filter

    for i = 1:size(images, 1)
        img = reshape(images(i, :), [27, 18]);
        filtered_img = filter2(B, double(img), 'same');
        output_images(i, :) = filtered_img(:)';
    end
end
