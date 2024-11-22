function ShowDetectionResult(Picture, Objects)

% Show the picture
figure, imshow(Picture), hold on;

% Define color coding for confidence levels
colours = ['b'; 'c'; 'm'; 'y'];
labels = {'High Confidence (> 0.8)', 'Moderate Confidence (> 0.5)', ...
          'Low Confidence (> 0.1)', 'Very Low Confidence (<= 0.1)'};

% Show the detected objects
if ~isempty(Objects)
    for n = 1:size(Objects, 1)
        % Extract bounding box coordinates
        x1 = Objects(n, 1);
        y1 = Objects(n, 2);
        width = Objects(n, 3);
        height = Objects(n, 4);
        x2 = x1 + width;
        y2 = y1 + height;
        
        % Compute confidence level
        confidence = Objects(n, 5); % Use the scaled confidence from the bbox
        
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

% Add legend with custom colors and multiline layout
legendLabels = {'\color{blue}High Confidence (> 0.8)', ...
                '\color{cyan}Moderate Confidence (> 0.5)', ...
                '\color{magenta}Low Confidence (> 0.1)', ...
                '\color{green}Very Low Confidence (<= 0.1)'};
legend(legendLabels, 'TextColor', 'black', 'FontSize', 12, 'Box', 'off', ...
    'Position', [0.1, 0.85, 0.2, 0.1], 'Orientation', 'vertical');

hold off;

end