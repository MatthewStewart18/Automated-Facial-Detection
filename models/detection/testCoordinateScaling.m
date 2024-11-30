function testCoordinateScaling(img)
    % Get original dimensions
    [origRows, origCols] = size(img);
    
    % Get point from original image
    figure('Name', 'Select Point');
    imshow(img);
    [x, y] = ginput(1);
    close;
    originalPoint = [x y];
    
    scales = [1.25, 1.5];
    figure('Name', 'Coordinate Scaling Test', 'Position', [100 100 1200 400]);
    
    % Original image
    subplot(1, 3, 1);
    imshow(img);
    hold on;
    plot(originalPoint(1), originalPoint(2), 'r.', 'MarkerSize', 20);
    title(sprintf('Original size: %dx%d', origRows, origCols));
    
    for i = 1:length(scales)
        % Scale image
        scaledImg = imresize(img, scales(i), 'bicubic');
        [newRows, newCols] = size(scaledImg);
        
        % Scale coordinates using dimension ratios
        rowRatio = newRows/origRows;
        colRatio = newCols/origCols;
        
        scaledPoint = [originalPoint(1)*colRatio, originalPoint(2)*rowRatio];
        
        % Plot
        subplot(1, 3, i+1);
        imshow(scaledImg);
        hold on;
        plot(scaledPoint(1), scaledPoint(2), 'r.', 'MarkerSize', 20);
        title(sprintf('Scale: %.2f\nSize: %dx%d', scales(i), newRows, newCols));
        
        fprintf('Original size: %dx%d, New size: %dx%d\n', origRows, origCols, newRows, newCols);
        fprintf('Scale %.2f - Original point: (%d,%d), Scaled point: (%d,%d)\n', ...
            scales(i), originalPoint(1), originalPoint(2), scaledPoint(1), scaledPoint(2));
    end
end