classdef ModelFactory
    properties
        Model                % Trained model
        ModelType            % Enum for model
        FeatureType          % Enum for feature extraction
        PreprocessingType    % Enum for preprocessing
        PreprocessingSteps   % Preprocessing steps
        FeatureExtractors    % Feature extraction steps
        Params               % Model parameters
        TrainingFunction     % Training function
        PredictionFunction   % Prediction function
        PCAParameters        % Struct to store PCA parameters (eigenVectors, meanX)
    end
    
    methods
        % Constructor
        function obj = ModelFactory(modelType, featureType, preprocessingType, params)
            obj.ModelType = modelType;
            obj.FeatureType = featureType;
            obj.Params = params;
            obj.PreprocessingType = preprocessingType;
            obj.FeatureExtractors = {};
            obj.PreprocessingSteps = {};
            obj.PCAParameters = struct();
            
            % Map ModelType to Training and Prediction Functions
            switch modelType
                case ModelType.SVM
                    obj.TrainingFunction = @SVMtraining;
                    obj.PredictionFunction = @extractPredictionsSVM;
                case ModelType.KNN
                    obj.TrainingFunction = @KNNtraining;
                    obj.PredictionFunction = @KNNTesting;
                case ModelType.LG
                    obj.TrainingFunction = @fitglm;
                    obj.PredictionFunction = @predict;
                otherwise
                    error('Unsupported ModelType');
            end
            
            % Map FeatureType to Feature Extraction Pipeline
            switch featureType
                case FeatureType.RawPix
                    extractor = struct('Function', {}, 'Args', {});
                    obj.FeatureExtractors{end + 1} = extractor;
                case FeatureType.Edges
                    extractor = struct('Function', @extractEdges);
                    extractor.Args = {};
                    obj.FeatureExtractors{end + 1} = extractor;
                case FeatureType.GaborPCA
                    extractor = struct('Function', @extractGabor);
                    extractor.Args = {};
                    obj.FeatureExtractors{end + 1} = extractor;
                    extractor = struct('Function', @extractPca);
                    extractor.Args = {100};
                    obj.FeatureExtractors{end + 1} = extractor;
                case FeatureType.EdgesPCA
                    extractor = struct('Function', @extractEdges);
                    extractor.Args = {};
                    obj.FeatureExtractors{end + 1} = extractor;
                    extractor = struct('Function', @extractPca);
                    extractor.Args = {};
                    obj.FeatureExtractors{end + 1} = extractor;
                case FeatureType.HOG
                    extractor = struct('Function', @extractHog);
                    extractor.Args = {};
                    obj.FeatureExtractors{end + 1} = extractor;
                case FeatureType.PCA
                    extractor = struct('Function', @extractPca);
                    extractor.Args = {};
                    obj.FeatureExtractors{end + 1} = extractor;
                otherwise
                    error('Unsupported FeatureType');
            end

            % Map PreprocessingType to Preprocessing Pipeline
            switch preprocessingType
                case PreprocessingType.None
                    pre = struct('Function', {}, 'Args', {});
                    obj.PreprocessingSteps{end + 1} = pre;
                case PreprocessingType.HistEq
                    pre = struct('Function', @histEq);
                    pre.Args = {};
                    obj.PreprocessingSteps{end + 1} = pre;
                otherwise
                    error('Unsupported FeatureType');
            end
        end
        
        % Add a preprocessing step
        function obj = addPreprocessingStep(obj, func, varargin)
            obj.PreprocessingSteps{end + 1} = struct('Function', func, 'Args', {varargin});
        end
        
        % Apply all preprocessing steps in sequential order
        function images = applyPreprocessing(obj, images)
            if obj.PreprocessingType ~= PreprocessingType.None
                for step = obj.PreprocessingSteps
                    func = step{1}.Function;
                    args = step{1}.Args;
                    images = preProcess(images, func, args{:});
                end
            end
        end
        
        % Add a feature extraction step
        function obj = addFeatureExtractionStep(obj, func, varargin)
            obj.FeatureExtractors{end + 1} = struct('Function', func, 'Args', {varargin});
        end
        
        % Apply feature extraction step
        function [features, obj] = applyFeatureExtraction(obj, images, isTraining)
            features = images;  
           
            % Apply all feature extraction functions in sequential order
            if obj.FeatureType ~= FeatureType.RawPix
                for fe = obj.FeatureExtractors
                    func = fe{1}.Function;
                    args = fe{1}.Args;
                    
                    % Special case for PCA, ensuring test set is projected
                    % to same vector space as training set
                    if isequal(func, @extractPca)
                        if isTraining
                            [eigenVectors, ~, meanX, Xpca] = func(features, args{:});
                            obj.PCAParameters.eigenVectors = eigenVectors;
                            obj.PCAParameters.meanX = meanX;
                            features = Xpca;
                        else
                            if isempty(obj.PCAParameters)
                                error('PCA parameters not found. Ensure the model is trained before testing.');
                            end
                            meanX = obj.PCAParameters.meanX;
                            eigenVectors = obj.PCAParameters.eigenVectors;
                            features = (features - meanX)*eigenVectors;
                        end
                        
                    % Apply same feature extraction for test and training 
                    % if not pca 
                    else
                        features = featureExtraction(features, func, args{:});
                    end
                end
            end
        end
        
        % Train the model
        function obj = train(obj, trainImages, trainLabels)
            % Apply preprocessing
            trainImages = obj.applyPreprocessing(trainImages);
            
            % Apply feature extraction
            [features, obj] = obj.applyFeatureExtraction(trainImages, true);
            
            % Train the model using the specified training function
            if isempty(obj.Params)
                obj.Model = obj.TrainingFunction(features, trainLabels);
            else
                obj.Model = obj.TrainingFunction(features, trainLabels, obj.Params);
            end
        end
        
        % Test the model
        function [predictions, confidence] = test(obj, testImages)
            % Apply preprocessing
            testImages = obj.applyPreprocessing(testImages);
            
            % Apply feature extraction
            [testFeatures, obj] = obj.applyFeatureExtraction(testImages, false);
            
            % Set default confidence if not supplied by model's test function
            defaultConfidence = 1;
            
            % Make predictions using the specified prediction function
            if obj.ModelType == ModelType.LG 
                predictions = obj.PredictionFunction(obj.Model, testFeatures);
                confidence = double(predictions);
                predictions = double(predictions >= 0.5);
            else
                if nargout(obj.PredictionFunction) > 1
                    [predictions, confidence] = obj.PredictionFunction(testFeatures, obj.Model);
                else
                    predictions = obj.PredictionFunction(testFeatures, obj.Model);
                    confidence = repmat(defaultConfidence, size(predictions)); % Assign default confidence
                end
            end
        end
        
        % Evaluate the model
        function evaluate(~, predictions, testLabels, testImages)
            fprintf('Evaluating model predictions...\n');
            [~] = calculateMetrics(predictions, testLabels);
            dispPreds(predictions, testLabels, testImages);
        end
    end
end
