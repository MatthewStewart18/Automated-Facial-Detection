function [Lut] = contrast_PL_LUT(gamma)
Lut = uint8(round(((0:255).^gamma) ./ (255.^(gamma-1))));
end