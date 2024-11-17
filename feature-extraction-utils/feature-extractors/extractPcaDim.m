function [reduced_features] = extractPcaDim(features, dim)
[~, score, ~] = pca(features);
reduced_features = score(:, 1:dim);
end

