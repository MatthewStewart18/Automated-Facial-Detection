function processedImages = preProcess(images, preprocessFunc, varargin)
    % `featureFunc` is a function handle, and `varargin` captures any extra parameters
    processedImages = preprocessFunc(images, varargin{:});
end

