function [predictions, windowPositions] = window(image, windowSize, stepSize, model)
    image = double(image);
    [rows, cols] = size(image);
    windowRows = windowSize(1);
    windowCols = windowSize(2);
    rowStep = stepSize(1);
    colStep = stepSize(2);

    outRows = ceil((rows - windowRows + 1) / rowStep);
    outCols = ceil((cols - windowCols + 1) / colStep);

    predictions = zeros(outRows*outCols, 2);
    windowPositions = zeros(outRows*outCols, 2);

    for outRow = 1:outRows
        for outCol = 1:outCols
            currentRow = (outRow - 1)*rowStep + 1;
            currentCol = (outCol - 1)*colStep + 1;
            window = image(currentRow:currentRow + windowRows - 1, currentCol:currentCol + windowCols - 1);
            window = reshape(window, 1, []);
            [prediction, confidence] = model.test(window);
            predictions((outRow*outCol-outCol)+outCol, :) = [prediction, confidence];
            windowPositions((outRow*outCol-outCol)+outCol, :) = [currentCol, currentRow];
        end
    end
end

