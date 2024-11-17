function features = featureExtraction(images, featureFunc, varargin)
    addpath ../../feature-extraction-utils/feature-extractors
    % `featureFunc` is a function handle, and `varargin` captures any extra parameters
    features = featureFunc(images, varargin{:});
end

