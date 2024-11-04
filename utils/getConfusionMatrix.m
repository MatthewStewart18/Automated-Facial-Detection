function [TP, TN, FP, FN] = getConfusionMatrix(predictions, labels)
    TP = sum(predictions == 1 & labels == 1);
    TP = TP(1);
    TN = sum(predictions == -1 & labels == -1);
    TN = TN(1);
    FP = sum(predictions == 1 & labels == -1);
    FP = FP(1);
    FN = sum(predictions == -1 & labels == 1);
    FN = FN(1);
    
    fprintf('\nDetailed Confusion Matrix:\n');
    fprintf('True Positives (TP): %d\n', TP);
    fprintf('True Negatives (TN): %d\n', TN);
    fprintf('False Positives (FP): %d\n', FP);
    fprintf('False Negatives (FN): %d\n', FN);
end

