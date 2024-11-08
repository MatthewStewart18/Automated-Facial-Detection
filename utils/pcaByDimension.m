function [reduced_features] = pcaByDimension(features, dim)
[~, score, ~] = pca(features);
reduced_features = score(:, 1:dim);
end

