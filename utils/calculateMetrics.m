function [accuracy, precision, recall, f1_score] = calculateMetrics(predictions, labels)
    [TP, TN, FP, FN] = getConfusionMatrix(predictions, labels);
    accuracy = (TP + TN) / length(labels) * 100;
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1_score = 2 * (precision * recall) / (precision + recall);

    fprintf('\nClassification Results:\n');
    fprintf('Accuracy: %.2f%%\n', accuracy);
    fprintf('Precision: %.2f\n', precision);
    fprintf('Recall: %.2f\n', recall);
    fprintf('F1 Score: %.2f\n', f1_score);
end