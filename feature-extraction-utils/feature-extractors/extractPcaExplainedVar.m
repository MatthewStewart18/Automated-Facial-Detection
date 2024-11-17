function [reducedFeatures, n_components] = extractPcaExplainedVar(features, varLevel)
[~, score, latent] = pca(features);
explained = cumsum(latent)./sum(latent);
n_components = find(explained >= varLevel, 1); % Keep <varLevel>% of variance
fprintf('Using %d PCA components\n', n_components);
reducedFeatures = score(:, 1:n_components);
end

