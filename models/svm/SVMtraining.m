function model = SVMtraining(images, labels, params)

    % Default parameters if fields are missing
    if ~isfield(params, 'lambda')
        params.lambda = 1e-20;
    end
    if ~isfield(params, 'C')
        params.C = Inf;
    end
    if ~isfield(params, 'kerneloption')
        params.kerneloption = 5;
    end
    if ~isfield(params, 'kernel')
        params.kernel = 'gaussian';
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