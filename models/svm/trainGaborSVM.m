function [model, accuracy, precision, recall, f1_score] = trainGaborSVM(gabor_features, train_labels)
    addpath SVM-KM
    
    fprintf('Processing optimized Gabor features...\n');
    
    % 1. Feature Preprocessing
    % Apply PCA to reduce dimensionality while preserving 95% of variance
    [coeff, score, latent] = pca(gabor_features, 'NumComponents', min(size(gabor_features,2), 500));
    explained = cumsum(latent) / sum(latent);
    n_components = find(explained >= 0.95, 1);
    reduced_features = score(:, 1:n_components);
    
    % 2. Class Balancing with SMOTE-like approach
    pos_idx = train_labels == 1;
    neg_idx = train_labels == -1;
    n_pos = sum(pos_idx);
    n_neg = sum(neg_idx);
    
    fprintf('\nOriginal class distribution - Positive: %d, Negative: %d\n', n_pos, n_neg);
    
    if n_pos < n_neg
        % Generate synthetic positive samples
        pos_samples = reduced_features(pos_idx, :);
        n_synthetic = n_neg - n_pos;
        synthetic_samples = zeros(n_synthetic, size(pos_samples, 2));
        
        for i = 1:n_synthetic
            % Select random sample
            idx = randi(n_pos);
            base_sample = pos_samples(idx, :);
            
            % Find k nearest neighbors
            k = 5;
            distances = pdist2(base_sample, pos_samples);
            [~, nn_indices] = sort(distances);
            nn_idx = nn_indices(randi([2, k+1]));
            
            % Generate synthetic sample
            diff = pos_samples(nn_idx, :) - base_sample;
            synthetic_samples(i, :) = base_sample + rand * diff;
        end
        
        % Combine original and synthetic samples
        balanced_features = [reduced_features; synthetic_samples];
        balanced_labels = [train_labels; ones(n_synthetic, 1)];
    else
        balanced_features = reduced_features;
        balanced_labels = train_labels;
    end
    
    % 3. SVM Training with optimized parameters
    fprintf('Training SVM with optimized parameters...\n');
    try
        % Adjusted parameters for better recall
        C = 10;        % Reduced cost parameter
        lambda = 1e-6; % Increased regularization
        sigma = 4.75;   % Adjusted kernel width
        
        % Train SVM
        [xsup, w, w0] = svmclass(balanced_features, balanced_labels, C, lambda, ...
            'gaussian', sigma, 1);
        
        model = struct();
        model.type = 'binary';
        model.xsup = xsup;
        model.w = w;
        model.w0 = w0;
        model.param.kernel = 'gaussian';
        model.param.sigmakernel = sigma;
        model.pca_coeff = coeff(:, 1:n_components);
        
        % Evaluate with relaxed confidence threshold
        fprintf('Evaluating model...\n');
        predictions = zeros(size(balanced_labels));
        confidences = zeros(size(balanced_labels));
        
        for i = 1:size(balanced_features, 1)
            [predictions(i), confidences(i)] = SVMTesting(balanced_features(i,:), model);
        end
        
        % Adjust confidence threshold to improve recall
        confidence_threshold = prctile(confidences, 30); % Increased from 5
        predictions(confidences < confidence_threshold) = -1;
        
        [accuracy, precision, recall, f1_score] = calculate_metrics(predictions, balanced_labels);
        print_results(accuracy, precision, recall, f1_score, predictions, balanced_labels);
        
    catch ME
        fprintf('Error in optimized Gabor SVM training: %s\n', ME.message);
        rethrow(ME);
    end
end

function [accuracy, precision, recall, f1_score] = calculate_metrics(predictions, labels)
    TP = sum(predictions == 1 & labels == 1);
    TN = sum(predictions == -1 & labels == -1);
    FP = sum(predictions == 1 & labels == -1);
    FN = sum(predictions == -1 & labels == 1);
    
    accuracy = (TP + TN) / length(labels) * 100;
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1_score = 2 * (precision * recall) / (precision + recall);
end

function print_results(accuracy, precision, recall, f1_score, predictions, labels)
    TP = sum(predictions == 1 & labels == 1);
    TN = sum(predictions == -1 & labels == -1);
    FP = sum(predictions == 1 & labels == -1);
    FN = sum(predictions == -1 & labels == 1);
    
    fprintf('\nOptimized Gabor SVM Results:\n');
    fprintf('Accuracy: %.2f%%\n', accuracy);
    fprintf('Precision: %.2f\n', precision);
    fprintf('Recall: %.2f\n', recall);
    fprintf('F1 Score: %.2f\n', f1_score);
    fprintf('\nConfusion Matrix:\n');
    fprintf('True Positives: %d\n', TP);
    fprintf('True Negatives: %d\n', TN);
    fprintf('False Positives: %d\n', FP);
    fprintf('False Negatives: %d\n', FN);
    fprintf('\nAdditional Metrics:\n');
    fprintf('Specificity: %.2f%%\n', (TN/(TN+FP))*100);
    fprintf('False Positive Rate: %.2f%%\n', (FP/(FP+TN))*100);
    fprintf('False Negative Rate: %.2f%%\n', (FN/(FN+TP))*100);
end