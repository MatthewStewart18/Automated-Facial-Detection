function [prediction, maxi] = SVMTesting(image, model)
    
    pred = svmval(image, model.xsup, model.w, model.w0, model.params.kernel, model.params.kerneloption);
    
    if pred > 0
        prediction = 1;
    else
        prediction = -1;
    end

    maxi = pred;
end