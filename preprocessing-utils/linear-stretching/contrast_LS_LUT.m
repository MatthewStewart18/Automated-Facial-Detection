function Lut = contrast_LS_LUT(m, c)
Lut = (0:255);
Lut(Lut < (-c)/m) = 0;                  
Lut(Lut > (255 - c)/m) = 255;
validEntries = Lut >= (-c)/m & Lut <= (255 - c)/m;
Lut(validEntries) = m * Lut(validEntries) + c;
Lut = uint8(Lut);
end