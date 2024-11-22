function [image] = histEq(images)
addpath ../../preprocessing-utils/hist-eq
image = zeros(size(images, 1), 27*18);
for i = 1:size(images, 1)
    img = reshape(images(i,:), [27, 18]);
    img = enhanceContrastHE(uint8(img));
    img = img(:)';
    image(i, :) = double(img);
end
end