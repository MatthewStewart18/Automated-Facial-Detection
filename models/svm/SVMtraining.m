function model = SVMtraining(images, labels)


    % first we check if the problem is binary classification or multiclass
    if max(labels)<2
        %binary classification
        model.type='binary';
    
        %SVM software requires labels -1 or 1 for the binary problem
        labels(labels==0)=-1;
    
        %Initilaise and setup SVM parameters
        lambda = 1e-20;
        C = inf;
        sigmakernel= 9.5;
        K=svmkernel(images,'gaussian',sigmakernel);
        kerneloption.matrix=K;
        kernel='numerical';
    
    
        % Calculate the support vectors
        [xsup,w,w0,pos,tps,alpha] = svmclass(images,labels,C,lambda,kernel,kerneloption,1);
    
        % create a structure encapsulating all teh variables composing the model
        model.xsup = xsup;
        model.w = w;
        model.w0 = w0;
    
        model.param.sigmakernel=sigmakernel;
        model.param.kernel=kernel;
    
    
    else
        %multiple class classification
         model.type='multiclass';
    
        %SVM software requires labels from 1 to N for the multi-class problem
        labels = labels+1;
        nbclass=max(labels);
    
        %Initilaise and setup SVM parameters
        lambda = 1e-5;
        C = 1;
        kerneloption = 0.5;
        kernel='gaussian';
    
        % Calculate the support vectors
        [xsup,w,b,nbsv]=svmmulticlassoneagainstall(images,labels,nbclass,C,lambda,kernel,kerneloption,1);
    
        % create a structure encapsulating all teh variables composing the model
        model.xsup = xsup;
        model.w = w;
        model.b = b;
        model.nbsv = nbsv;
    
        model.param.kerneloption=kerneloption;
        model.param.kernel=kernel;
    
    end
    
    
    
    end%