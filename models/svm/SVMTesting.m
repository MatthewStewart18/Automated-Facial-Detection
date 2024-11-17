function [prediction, confidence] = SVMTesting(image, model)

    if strcmp(model.type, 'binary')
        % Binary SVM case
        % kerneloption.matrix = svmkernel(image, 'gaussian', model.param.sigmakernel, model.xsup);
        pred = svmval(image, model.xsup, model.w, model.w0, model.param.kernel, model.param.kerneloption);
        
        % Determine prediction and confidence
        if pred > 0
            prediction = 1;
        else
            prediction = -1;
        end
        confidence = abs(pred);  % Confidence as the distance from the decision boundary
    
    else
        % Multi-class SVM case
        [pred, maxi] = svmmultival(image, model.xsup, model.w, model.b, model.nbsv, model.param.kernel, model.param.kerneloption);
        
        prediction = round(pred) - 1;
        confidence = abs(maxi);  % Use maxi as the confidence measure for multi-class
    end

end