function [model, accuracy, precision, recall, f1_score] = trainGaborPCASVM(score, train_labels)
    addpath SVM-KM
    % 1. Advanced Feature Engineering
    % Standardize and add polynomial features
    score_std = normalize(score, 'zscore');
    
    % Add squared terms for non-linear features
    score_squared = score_std.^2;
    SVMtraining_v2()
    
    % Add interaction terms between key components
    n_components = min(5, size(score_std, 2)); % Use top 5 components for interactions
    interaction_terms = zeros(size(score_std,1), nchoosek(n_components,2));
    idx = 1;
    for i = 1:n_components
        for j = (i+1):n_components
            interaction_terms(:,idx) = score_std(:,i) .* score_std(:,j);
            idx = idx + 1;
        end
    end
    
    % Combine all features
    score_enhanced = [score_std, score_squared(:,1:n_components), interaction_terms];
    
    % 2. Advanced Feature Selection
    % Remove highly correlated features
    correlation_matrix = corrcoef(score_enhanced);
    [row, col] = find(triu(abs(correlation_matrix) > 0.95, 1));
    features_to_remove = unique(col);
    score_enhanced(:, features_to_remove) = [];
    
    % 3. Robust Scaling with Outlier Handling
    % Use robust scaling (based on quartiles)
    for i = 1:size(score_enhanced, 2)
        q1 = prctile(score_enhanced(:,i), 25);
        q3 = prctile(score_enhanced(:,i), 75);
        iqr = q3 - q1;
        if iqr > 0
            score_enhanced(:,i) = (score_enhanced(:,i) - q1) / iqr;
        end
    end
    
    % 4. Advanced Class Balancing with Clustering-based SMOTE
    pos_idx = find(train_labels == 1);
    neg_idx = find(train_labels == -1);
    n_pos = length(pos_idx);
    n_neg = length(neg_idx);
    
    fprintf('Original distribution - Positive: %d, Negative: %d\n', n_pos, n_neg);
    
    if n_pos ~= n_neg
        % Determine minority and majority class
        if n_pos < n_neg
            minority_idx = pos_idx;
            majority_idx = neg_idx;
            minority_label = 1;
        else
            minority_idx = neg_idx;
            majority_idx = pos_idx;
            minority_label = -1;
        end
        
        % Cluster majority class
        n_clusters = min(5, length(majority_idx));
        try
            [idx_clusters, centroids] = kmeans(score_enhanced(majority_idx,:), n_clusters, ...
                'MaxIter', 100, 'Replicates', 3);
            
            % Generate synthetic samples near cluster boundaries
            n_synthetic = abs(n_pos - n_neg);
            synthetic_samples = zeros(n_synthetic, size(score_enhanced, 2));
            synthetic_labels = minority_label * ones(n_synthetic, 1);
            
            for i = 1:n_synthetic
                % Select random minority sample
                base_idx = minority_idx(randi(length(minority_idx)));
                % Find nearest cluster centroid
                [~, nearest_cluster] = min(pdist2(score_enhanced(base_idx,:), centroids));
                
                % Generate synthetic sample
                noise_scale = 0.1;
                synthetic_samples(i,:) = score_enhanced(base_idx,:) + ...
                    noise_scale * randn(1, size(score_enhanced, 2));
            end
            
            % Combine with original data
            score_enhanced = [score_enhanced; synthetic_samples];
            train_labels = [train_labels; synthetic_labels];
        catch
            fprintf('Clustering failed, using simple oversampling...\n');
        end
    end
    
    % 5. Optimized SVM Training
    fprintf('Training enhanced SVM model...\n');
    try
        % Optimized hyperparameters based on face detection task
        C = 15;           % Increased cost parameter
        lambda = 1e-6;    % Refined regularization
        sigma = 1.5;      % Optimized kernel width
        
        % Train SVM
        [xsup, w, w0] = svmclass(score_enhanced, train_labels, C, lambda, ...
            'gaussian', sigma, 1);
        
        % Create enhanced model
        model = struct();
        model.type = 'binary';
        model.xsup = xsup;
        model.w = w;
        model.w0 = w0;
        model.param.kernel = 'gaussian';
        model.param.sigmakernel = sigma;
        
        % Additional model parameters for feature processing
        model.feature_params.n_components = n_components;
        model.feature_params.removed_features = features_to_remove;
        
        % Make predictions with confidence thresholding
        fprintf('Evaluating model...\n');
        predictions = zeros(size(train_labels));
        confidences = zeros(size(train_labels));
        
        for i = 1:length(train_labels)
            [pred, conf] = SVMTesting(score_enhanced(i,:), model);
            predictions(i) = pred;
            confidences(i) = conf;
        end
        
        % Apply confidence thresholding
        confidence_threshold = prctile(confidences, 20);  % Adjust based on distribution
        low_confidence = confidences < confidence_threshold;
        predictions(low_confidence) = -1;  % Default to negative for low confidence
        
        % Calculate final metrics
        [accuracy, precision, recall, f1_score] = calculateMetrics(predictions, train_labels);
        
        % Print detailed results
        printResults(accuracy, precision, recall, f1_score, predictions, train_labels);
        
    catch ME
        fprintf('Error in enhanced training: %s\n', ME.message);
        rethrow(ME);
    end
end

function [accuracy, precision, recall, f1_score] = calculateMetrics(predictions, labels)
    TP = sum(predictions == 1 & labels == 1);
    TN = sum(predictions == -1 & labels == -1);
    FP = sum(predictions == 1 & labels == -1);
    FN = sum(predictions == -1 & labels == 1);
    
    accuracy = (TP + TN) / length(labels) * 100;
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1_score = 2 * (precision * recall) / (precision + recall);
end

function printResults(accuracy, precision, recall, f1_score, predictions, labels)
    TP = sum(predictions == 1 & labels == 1);
    TN = sum(predictions == -1 & labels == -1);
    FP = sum(predictions == 1 & labels == -1);
    FN = sum(predictions == -1 & labels == 1);
    
    fprintf('\nEnhanced SVM Classification Results:\n');
    fprintf('Accuracy: %.2f%%\n', accuracy);
    fprintf('Precision: %.2f\n', precision);
    fprintf('Recall: %.2f\n', recall);
    fprintf('F1 Score: %.2f\n', f1_score);
    fprintf('\nDetailed Confusion Matrix:\n');
    fprintf('True Positives (TP): %d\n', TP);
    fprintf('True Negatives (TN): %d\n', TN);
    fprintf('False Positives (FP): %d\n', FP);
    fprintf('False Negatives (FN): %d\n', FN);
    fprintf('\nAdditional Metrics:\n');
    fprintf('Specificity: %.2f%%\n', (TN/(TN+FP))*100);
    fprintf('Balanced Accuracy: %.2f%%\n', ((TP/(TP+FN) + TN/(TN+FP))/2)*100);
end