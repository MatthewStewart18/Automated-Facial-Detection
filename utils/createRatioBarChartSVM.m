function createRatioBarChartSVM(confusionMatrix, modelName, accuracy, precision, recall, f1_score)
    figure('Position', [100 100 1200 500]);

    accuracy = accuracy / 100;
    
    % First subplot - Bar chart for metrics
    subplot(1, 2, 1);
    metrics = [accuracy, precision, recall, f1_score];
    colors = [0.2 0.7 0.2;  % Green for Accuracy
              0.2 0.2 0.7;  % Blue for Precision
              0.7 0.2 0.2;  % Red for Recall
              0.7 0.5 0.2]; % Orange for F1
    
    b = bar(metrics, 'FaceColor', 'flat');
    
    % Set colors
    for j = 1:4
        b.CData(j,:) = colors(j,:);
    end
    
    title(sprintf('SVM %s Metrics', modelName), 'FontSize', 12);
    ylabel('Score');
    set(gca, 'XTickLabel', {'Accuracy', 'Precision', 'Recall', 'F1-Score'});
    ylim([0 1]); % Scale from 0 to 1
    
    % labels
    for i = 1:length(metrics)
        text(i, metrics(i), ...
             sprintf('%.3f', metrics(i)), ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom');
    end
    grid on;
    
    % Second subplot - Confusion Matrix
    subplot(1, 2, 2);
    [TP, FP, FN, TN] = deal(confusionMatrix(1), confusionMatrix(2), confusionMatrix(3), confusionMatrix(4));
    
    
    green = [144/255, 173/255, 116/255]; % correct predictions
    blue = [197/255, 216/255, 226/255];   % incorrect predictions
    
    % Create patches for each cell with flipped vertical orientation
    hold on;
    % TP
    patch([0 1 1 0], [1 1 2 2], green);
    % FP
    patch([1 2 2 1], [1 1 2 2], blue);
    % FN
    patch([0 1 1 0], [0 0 1 1], blue);
    % TN
    patch([1 2 2 1], [0 0 1 1], green);
    
    % Add text annotations with flipped positions
    text(0.5, 1.5, sprintf('TP\n%d', TP), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontWeight', 'bold');
    text(1.5, 1.5, sprintf('FP\n%d', FP), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontWeight', 'bold');
    text(0.5, 0.5, sprintf('FN\n%d', FN), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontWeight', 'bold');
    text(1.5, 0.5, sprintf('TN\n%d', TN), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontWeight', 'bold');
    
    % Set axis properties
    axis([0 2 0 2]);
    set(gca, 'XTick', [0.5 1.5], 'YTick', [0.5 1.5]);
    set(gca, 'XTickLabel', {'Positive', 'Negative'}, 'FontWeight', 'bold');
    set(gca, 'YTickLabel', {'Negative', 'Positive'}, 'FontWeight', 'bold'); % Flipped Y labels
    xlabel('Predicted', 'FontWeight', 'bold');
    ylabel('Actual', 'FontWeight', 'bold');
    title('Confusion Matrix', 'FontWeight', 'bold');
    
    sgtitle(sprintf('Performance Analysis - %s', modelName), 'FontSize', 14);
    
    % background white
    set(gcf, 'Color', 'white');
end