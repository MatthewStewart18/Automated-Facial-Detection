function [predictions, windowPositions] = window(image, windowSize, stepSize, model)
    image = double(image);
    [rows, cols] = size(image);

    % extract the window size and step sizes
    windowRows = windowSize(1);
    windowCols = windowSize(2);
    rowStep = stepSize(1);
    colStep = stepSize(2);
    
    % calculate the rows and cols we will iterate over
    outRows = ceil((rows - windowRows + 1) / rowStep);
    outCols = ceil((cols - windowCols + 1) / colStep);
    
    % initialise final predictions and positions 
    predictions = zeros(outRows*outCols, 2);
    windowPositions = zeros(outRows*outCols, 2);
    
    index = 0;
    for outRow = 1:outRows
        for outCol = 1:outCols
        % Compute top-left corner of the current window
        currentRow = (outRow - 1) * rowStep + 1;
        currentCol = (outCol - 1) * colStep + 1;

        % Check if the window exceeds image bounds
        if currentRow + windowRows - 1 > size(image, 1) || currentCol + windowCols - 1 > size(image, 2)
            continue;
        end

        % Extract the current window
        window = image(currentRow:currentRow + windowRows - 1, currentCol:currentCol + windowCols - 1);
        window = reshape(window, 1, []);

        % Get model predictions
        [prediction, confidence] = model.test(window);

        % Store predictions and window positions
        index = index + 1;
        if numel(confidence) == 2 % Check if confidence has two elements
            confidence = max(confidence); % Take the greater value
        end
        predictions(index, :) = [prediction, confidence];
        windowPositions(index, :) = [currentCol, currentRow];
        end
    end
    
end

