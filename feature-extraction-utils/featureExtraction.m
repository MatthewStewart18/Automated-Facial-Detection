function varargout = featureExtraction(images, featureFunc, varargin)
    % `featureFunc` is a function handle, and `varargin` captures any extra parameters
    [varargout{1:nargout}] = featureFunc(images, varargin{:});
end

