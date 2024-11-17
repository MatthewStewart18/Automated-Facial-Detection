function varargout = featureExtraction(images, featureFunc, varargin)
    addpath ../../feature-extraction-utils/feature-extractors
    % `featureFunc` is a function handle, and `varargin` captures any extra parameters
    [varargout{1:nargout}] = featureFunc(images, varargin{:});
end

