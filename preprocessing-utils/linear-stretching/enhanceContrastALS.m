function enhancedContrastALS = enhanceContrastALS(Iin)
    histImage = myHistogram(Iin);
    binsGreater10 = find(histImage > 5);
    i1 = binsGreater10(1);
    lastIndex = length(binsGreater10);
    i2 = binsGreater10(lastIndex);
    orginalDifferenceInDynamicRange = i2-i1;
    m = 255/orginalDifferenceInDynamicRange;
    c = -m * i1;
    LUT = contrast_LS_LUT(m,c);
    resultImage = intlut(Iin, LUT);
   
enhancedContrastALS = resultImage;
        
% function [Iout, Lut] = enhanceContrastALS(Iin)
% [m, c] = findOptimalLS(myHistogram(Iin));
% Lut = contrast_LS_LUT(m, c);
% Iout = intlut(Iin,Lut);
% end