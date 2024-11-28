function visualiseResults(testImage, detections, groundTruthPath)
    % Read ground truth
    gt_file = fopen(groundTruthPath, 'r');
    gt_data = textscan(gt_file, '%d %d %d %d', 'CommentStyle', '#');
    fclose(gt_file);
    groundTruth = [gt_data{1}, gt_data{2}, gt_data{3}, gt_data{4}];
    
    % Evaluate and print metrics
    [accuracy, precision, recall, cumtp, cumfp, fn] = evaluateDetections(detections, groundTruth, 0.6);
    fprintf('Accuracy: %.3f, Precision: %.3f, Recall: %.3f\n, TP: %.3f, FP: %.3f, FN: %.3fd\n', ...
        accuracy, precision(end), recall(end), round(cumtp(end)), round(cumfp(end), fn));
    
    imshow(testImage, []);
    hold on;

    % Plot all ground truth boxes first (in green)
    for i = 1:size(groundTruth, 1)
        rectangle('Position', groundTruth(i,:), 'EdgeColor', 'g', 'LineWidth', 2);
    end
    
    % Then plot detections and calculate IoU
    for i = 1:size(detections, 1)
        rectangle('Position', detections(i,1:4), 'EdgeColor', 'r', 'LineWidth', 2);
        
        % Find best IoU with ground truth boxes
        maxIoU = 0;
        for j = 1:size(groundTruth, 1)
            iou = bboxOverlapRatio(detections(i,1:4), groundTruth(j,:));
            if iou > maxIoU
                maxIoU = iou;
            end
        end
        
        % Display IoU score
        text(detections(i,1), detections(i,2)-5, sprintf('IoU: %.2f', maxIoU), ...
            'Color', 'r', 'FontWeight', 'bold', 'BackgroundColor', 'white');
    end

    h1 = plot(NaN,NaN,'r-', 'LineWidth', 2);
    h2 = plot(NaN,NaN,'g-', 'LineWidth', 2);
    legend([h1, h2], 'Detection', 'Ground Truth', 'Location', 'southoutside');
    % set(lgd, 'EdgeColor', 'black', ...        
    %          'Box', 'on', ...                 
    %          'BackgroundColor', 'white', ...
    %          'Position', [0.3, 0.1, 0.2, 0.1]); 

    % Create metrics text box
    metrics_str = sprintf(['Accuracy: %.3f\n' ...
                          'Recall: %.3f\n' ...
                          'Precision: %.3f\n' ...
                          'TP: %d\n' ...
                          'FP: %d\n' ...
                          'FN: %d'], ...
                          accuracy, recall(end), precision(end), ...
                          round(cumtp(end)), round(cumfp(end)), fn);
    
    % Get current axes position
    axpos = get(gca, 'Position');

    % Create new axes for text, positioned to the right of the legend
    annotation('textbox', ...
              [axpos(1)+axpos(3)*0.7, axpos(2)+0.2, 0.2, 0.1], ... % [x, y, width, height]
              'String', metrics_str, ...
              'FitBoxToText', 'on', ...
              'BackgroundColor', 'white', ...
              'EdgeColor', 'black', ...
              'HorizontalAlignment', 'left');
    title('Detection Results with IoU Scores');
    hold off;
end
