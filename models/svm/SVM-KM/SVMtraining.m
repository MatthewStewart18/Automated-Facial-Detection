function model = SVMtraining(images, labels, params)
    % Check if params was provided; if not, initialize with defaults
    if nargin < 3
        params.lambda = 1e-20;         % Regularization parameter
        params.C = Inf;                % Trade-off parameter for SVM
        params.kerneloption = 4;       % Kernel-specific parameter (e.g., Gaussian width)
        params.kernel = 'poly';    % Type of kernel (e.g. 'poly', 'gaussian')
    end

    % Use default parameters for missing fields
    if ~isfield(params, 'lambda')
        params.lambda = 1e-20;
    end
    if ~isfield(params, 'C')
        params.C = Inf;
    end
    if ~isfield(params, 'kerneloption')
        params.kerneloption = 4;
    end
    if ~isfield(params, 'kernel')
        params.kernel = 'poly';
    end

    %binary classification
    model.type='binary';

    %SVM software requires labels -1 or 1 for the binary problem
    labels(labels==0)=-1;

    % Calculate the support vectors
    [xsup,w,w0,pos,tps,alpha] = svmclass(images, labels, params.C, params.lambda, ...
                                            params.kernel, params.kerneloption, 1);

    % create a structure encapsulating all the variables composing the model
    model.xsup = xsup;
    model.w = w;
    model.w0 = w0;
    model.params=params;   
end