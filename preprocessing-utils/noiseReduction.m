function [Iout] = noiseReduction(I,N)
B = ones(N, N)/(N^2);
Iout = filter2(double(B), double(I));
end