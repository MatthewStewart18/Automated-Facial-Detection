function [image] = powerLaw(images, gamma)
addpath ../../preprocessing-utils/power-law
image = zeros(size(images, 1), 27*18);
for i = 1:size(images, 1)
    img = reshape(images(i,:), [27, 18]);
    img = enhanceContrastPL(uint8(img), gamma);
    img = img(:)';
    image(i, :) = double(img);
end
end