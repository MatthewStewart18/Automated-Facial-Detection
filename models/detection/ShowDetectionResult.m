function ShowDetectionResult(Picture, bboxs)

% Show the picture
figure, imshow(Picture),hold on;

% Define color coding for confidence levels
colours = ['b'; 'c'; 'm'; 'y'];

% Show the detected objects
if ~isempty(bboxs)
    for n = 1:size(bboxs, 1)
        % Extract bounding box coordinates
        x1 = bboxs(n, 1);
        y1 = bboxs(n, 2);
        width = bboxs(n, 3);
        height = bboxs(n, 4);
        x2 = x1 + width;
        y2 = y1 + height;
        
        % Compute confidence level
        confidence = bboxs(n, 5); % Use the scaled confidence from the bbox
        
        if confidence > 0.8
            c = 1; % High confidence
        elseif confidence > 0.5
            c = 2; % Moderate confidence
        elseif confidence > 0.1
            c = 3; % Low confidence
        else
            c = 4; % Very low confidence
        end
        
        % Draw bounding box
        plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], colours(c), 'LineWidth', 2);
        
        % Add confidence score as a label close to the top-left corner of the box
        text(x1 + 2, y1 - 5, sprintf('%.2f', confidence), 'Color', colours(c), ...
            'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'white', ...
            'EdgeColor', 'black');
    end
end

hold off;

end