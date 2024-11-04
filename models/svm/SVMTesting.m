function [prediction, confidence] = SVMTesting(image, model)
    if strcmp(model.type, 'binary')
        % Calculate kernel matrix
        K = svmkernel(image, 'gaussian', model.param.sigmakernel, model.xsup);
        
        % Get raw prediction score
        pred_score = K * model.w + model.w0;
        
        % Calculate confidence as distance from decision boundary
        confidence = abs(pred_score);
        
        % Apply sigmoid scaling to get probability-like confidence
        confidence = 1 / (1 + exp(-2 * confidence));
        
        % Make prediction with enhanced decision boundary
        if pred_score > 0
            prediction = 1;
        else
            prediction = -1;
        end
    else
        % For multiclass (not used here)
        [prediction, confidence] = svmmultival(image, model.xsup, model.w, model.b, ...
            model.nbsv, model.param.kernel, model.param.sigmakernel);
    end
end