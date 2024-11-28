function visualizeResults(testImage, detections, groundTruthPath)
% Read ground truth
gt_file = fopen(groundTruthPath, 'r');
gt_data = textscan(gt_file, '%d %d %d %d', 'CommentStyle', '#');
fclose(gt_file);
groundTruth = [gt_data{1}, gt_data{2}, gt_data{3}, gt_data{4}];

% Evaluate and print metrics
[precision, recall, cumtp, cumfp] = evaluateDetections(detections, groundTruth, 0.6);
fprintf('Precision: %.3f, Recall: %.3f\n, TP: %.3f, FP: %.3f\n', ...
    precision(end), recall(end), cumtp(end), cumfp(end));

% Visualize comparison
% figure(1);
% title('Ground Truth (Green) vs Detections (Red)');
% figure('Name', 'Detection Results');
imshow(testImage, []);
hold on;
    for i = 1:size(detections, 1)
        rectangle('Position', detections(i,1:4), 'EdgeColor', 'r', 'LineWidth', 2);
    
        % Find best IoU with ground truth boxes
        maxIoU = 0;
        bestMatch = 0;
        for j = 1:size(groundTruth, 1)
            iou = bboxOverlapRatio(detections(i,1:4), groundTruth(j,:));
            if  iou > maxIoU
                maxIoU = iou;
                bestMatch = j;
            % maxIoU = max(maxIoU, iou);
            end
        end
        if maxIoU > 0
            rectangle('Position', groundTruth(bestMatch,:), 'EdgeColor', 'g', 'LineWidth', 2);
        end
    
        % Display IoU score
        text(detections(i,1), detections(i,2)-5, sprintf('IoU: %.2f', maxIoU), ...
             'Color', 'r', 'FontWeight', 'bold', 'BackgroundColor', 'white');
    end

    h1 = plot(NaN,NaN,'r-', 'LineWidth', 2);
    h2 = plot(NaN,NaN,'g-', 'LineWidth', 2);
    legend([h1, h2], 'Detection', 'Ground Truth', 'Location', 'southoutside');
    title('Detection Results with IoU Scores');
    hold off;
end