function [m,c] = findOptimalLS(arr)

no_noise = find(arr>10);
delta_O = 255-0;
delta_I = no_noise(end)-no_noise(1);

m = (delta_O)/(delta_I);
c = -1 * m * no_noise(1);

end