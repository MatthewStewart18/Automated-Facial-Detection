function [Lut] = brightnessLUT(c)
Lut = (1:256)-1;
Lut = Lut + c;
Lut(Lut < 0) = 0;
Lut(Lut > 255) = 255;
Lut=uint8(Lut);
end