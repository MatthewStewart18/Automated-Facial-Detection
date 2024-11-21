classdef ModelFactory
    properties
        Model                % Trained model
        ModelType            % Enum for model type (e.g., SVM, KNN)
        FeatureType          % Enum for feature extraction type
        PreprocessingSteps   % Preprocessing steps
        FeatureExtractors    % Feature extraction steps
        Params               % Model parameters
        TrainingFunction     % Training function
        PredictionFunction   % Prediction function
    end
    
    methods
        % Constructor
        function obj = ModelFactory(modelType, featureType, params)
            obj.ModelType = modelType;
            obj.FeatureType = featureType;
            obj.Params = params;
            obj.FeatureExtractors = {};
            obj.PreprocessingSteps = {};
            
            % Map ModelType to Training and Prediction Functions
            switch modelType
                case ModelType.SVM
                    obj.TrainingFunction = @SVMtraining;
                    obj.PredictionFunction = @extractPredictionsSVM;
                case ModelType.KNN
                    obj.TrainingFunction = @KNNtraining;
                    obj.PredictionFunction = @KNNTesting;
                otherwise
                    error('Unsupported ModelType');
            end
            
            % Map FeatureType to Feature Extraction Pipeline
            switch featureType
                case FeatureType.Edges
                    extractor = struct('Function', @extractEdges);
                    extractor.Args = {};
                    obj.FeatureExtractors{end + 1} = extractor;
                case FeatureType.Gabor
                    extractor = struct('Function', @extractGabor);
                    extractor.Args = {};
                    obj.FeatureExtractors{end + 1} = extractor;
                case FeatureType.PCA
                    extractor = struct('Function', @extractPcaExplainedVar);
                    extractor.Args = {0.95};
                    obj.FeatureExtractors{end + 1} = extractor;
                otherwise
                    error('Unsupported FeatureType');
            end
        end
        
        % Add preprocessing step
        function obj = addPreprocessingStep(obj, func, varargin)
            obj.PreprocessingSteps{end + 1} = struct('Function', func, 'Args', {varargin});
        end
        
        % Apply preprocessing step
        function images = applyPreprocessing(obj, images)
            for step = obj.PreprocessingSteps
                func = step{1}.Function;
                args = step{1}.Args;
                images = preProcess(images, func, args{:});
            end
        end
        
        % Add feature extraction step
        function obj = addFeatureExtractionStep(obj, func, varargin)
            obj.FeatureExtractors{end + 1} = struct('Function', func, 'Args', {varargin});
        end
        
        % Apply feature extraction step
        function features = applyFeatureExtraction(obj, images)
            features = images;  % Start with raw images
            
            % Apply all feature extraction functions in sequence
            for fe = obj.FeatureExtractors
                func = fe{1}.Function;
                args = fe{1}.Args;
                features = featureExtraction(features, func, args{:});
            end
        end
        
        % Train the model
        function obj = train(obj, trainImages, trainLabels)
            % Apply preprocessing
            trainImages = obj.applyPreprocessing(trainImages);
            
            % Apply feature extraction
            features = obj.applyFeatureExtraction(trainImages);
            
            % Train the model using the specified training function
            obj.Model = obj.TrainingFunction(features, trainLabels, obj.Params);
        end
        
        % Test the model
        function predictions = test(obj, testImages)
            % Apply preprocessing
            testImages = obj.applyPreprocessing(testImages);
            
            % Apply feature extraction
            testFeatures = obj.applyFeatureExtraction(testImages);
            
            % Make predictions using the specified prediction function
            predictions = obj.PredictionFunction(testFeatures, obj.Model);
        end
        
        % Evaluate the model
        function evaluate(~, predictions, testLabels, testImages)
            fprintf('Evaluating model predictions...\n');
            [~] = calculateMetrics(predictions, testLabels);
            dispPreds(predictions, testLabels, testImages);
        end
    end
end
