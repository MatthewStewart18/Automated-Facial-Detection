function dEuc = EuclideanDistance(sample1, sample2)
    dEuc = sqrt(sum((double(sample1) - double(sample2)).^ 2, 2));
end
