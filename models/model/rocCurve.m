function rocCurve(test_labels, confidences)
    % Compute the ROC curve
    [fpr, tpr, ~, auc] = perfcurve(test_labels, confidences, 1);
    
    % Plot the ROC curve
    figure;
    plot(fpr, tpr, 'LineWidth', 2);
    xlabel('False Positive Rate (FPR)');
    ylabel('True Positive Rate (TPR)');
    title(['ROC Curve (AUC = ' num2str(auc) ')']);
    grid on;
end

