function [predictions, windowPositions] = windowNN(image, windowSize, stepSize, net)
    image = double(image);
    [rows, cols] = size(image);

    % Extract window size and step sizes
    windowRows = windowSize(1);
    windowCols = windowSize(2);
    rowStep = stepSize(1);
    colStep = stepSize(2);

    % Calculate the number of windows
    outRows = ceil((rows - windowRows + 1) / rowStep);
    outCols = ceil((cols - windowCols + 1) / colStep);

    % Initialize predictions and positions
    predictions = zeros(outRows * outCols, 2);
    windowPositions = zeros(outRows * outCols, 2);

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

            % Extract the current window and reshape for prediction
            window = image(currentRow:currentRow + windowRows - 1, currentCol:currentCol + windowCols - 1);
            window = reshape(window, 1, []); % Flatten to a vector
            
            % Predict using the neural network
            confidence = net(window'); % Transpose for correct input format
            
            % Store predictions and window positions
            index = index + 1;
            predictions(index, :) = [confidence >= 0.5, confidence]; % Threshold at 0.5 for binary classification
            windowPositions(index, :) = [currentCol, currentRow];
        end
    end

    % Trim unused preallocated space
    predictions = predictions(1:index, :);
    windowPositions = windowPositions(1:index, :);
end