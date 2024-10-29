function [Lut] = contrast_HE_LUT(Iin)
histCounts = imhist(Iin, 256); 
CH = cumsum(histCounts); 
numPixels = (numel(Iin));
Lut = uint8(max(0, round(255 * CH/numPixels)));
end