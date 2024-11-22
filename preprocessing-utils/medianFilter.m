function [output_images] = medianFilter(images, N)
    output_images = zeros(size(images, 1), 27*18);

    for i = 1:size(images, 1)
        img = reshape(images(i, :), [27, 18]);
        filtered_img = medfilt2(img, [N, N]);
        output_images(i, :) = filtered_img(:)';
    end
end
