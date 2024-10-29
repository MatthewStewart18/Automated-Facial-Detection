function [Iout, Lut] = enhanceContrastALS(Iin)
[m, c] = findOptimalLS(myHistogram(Iin));
Lut = contrast_LS_LUT(m, c);
Iout = intlut(Iin,Lut);
end