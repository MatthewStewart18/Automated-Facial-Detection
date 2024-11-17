function [prediction, confidence] = SVMTesting(image, model)
    % Binary SVM case
    pred = svmval(image, model.xsup, model.w, model.w0, model.params.kernel, model.params.kerneloption);
    
    % Determine prediction and confidence
    if pred > 0
        prediction = 1;
    else
        prediction = -1;
    end
    confidence = abs(pred);  % Confidence as the distance from the decision boundary

end