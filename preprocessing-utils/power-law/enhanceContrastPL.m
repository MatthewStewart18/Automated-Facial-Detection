function [Iout] = enhanceContrastPL(Iin,gamma)
Lut = contrast_PL_LUT(gamma);
Iout = intlut(Iin,Lut);
end