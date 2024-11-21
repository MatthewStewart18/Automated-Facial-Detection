classdef ModelFactory
    properties
        Model                 % Trained model (SVM, k-NN, etc.)
        PreprocessingSteps    % Cell array of preprocessing steps
        FeatureExtractors     % Cell array of feature extraction steps (functions + arguments)
        TrainingFunction      % Handle to model training function
        PredictionFunction    % Handle to prediction function
        Params                % Parameters for the training function
    end
    
    methods
        % Constructor
        function obj = ModelFactory(trainingFunc, predictionFunc, params)
            if nargin > 0
                obj.TrainingFunction = trainingFunc;
                obj.PredictionFunction = predictionFunc;
                obj.Params = params;
                obj.FeatureExtractors = {};  
                obj.PreprocessingSteps = {};
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
