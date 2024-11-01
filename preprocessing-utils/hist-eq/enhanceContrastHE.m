function [Iout] = enhanceContrastHE(Iin)
disp(Iin)
histCounts = imhist(Iin, 256); 
CH = cumsum(histCounts); 

% Normalize the cumulative histogram (CDF) to range [0, 1]
CH_normalized = CH / max(CH); 

Iout = uint8(CH_normalized(double(Iin) + 1) * 255);
end
