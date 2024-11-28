function visualizeLBPGrid(image)
    % Apply adaptive histogram equalization
    
    % Set fixed cell size to [6,6]
    cellSize = [6 6];
    
    % Get LBP features
    lbpFeatures = extractLBPFeatures(image, ...
        'CellSize', cellSize, ...
        'Radius', 1, ...
        'NumNeighbors', 8);
    
    % Calculate number of cells
    numCellsY = floor(size(image,1)/cellSize(1));  % 4 for 27/6
    numCellsX = floor(size(image,2)/cellSize(2));  % 3 for 18/6
    
    % Create weight matrix adjusted for 4x3 grid
    weights = ones(numCellsY, numCellsX);  % Base weight of 1
    
    % Adjust weights for smaller grid
    % Eye region weights
    weights(1:2, :) = 1.5;  % Upper face area
    % Center weights for nose/mouth
    weights(2:3, 2) = 1.3;
    
    figure('Position', [100 100 800 400]);
    
    % Original image with grid
    subplot(1, 2, 1);
    imshow(image, [], 'InitialMagnification', 'fit');
    hold on;
    
    % Draw grid lines
    for i = 1:numCellsX
        x = i * cellSize(2);
        line([x x], [0 size(image,1)], 'Color', 'k', 'LineWidth', 1);
    end
    for i = 1:numCellsY
        y = i * cellSize(1);
        line([0 size(image,2)], [y y], 'Color', 'k', 'LineWidth', 1);
    end
    title('Face Image with 6x6 Cells');
    
    % LBP pattern visualization with weights
    subplot(1, 2, 2);
    featuresPerCell = size(lbpFeatures, 2) / (numCellsY * numCellsX);
    lbpPatterns = reshape(lbpFeatures, [numCellsY, numCellsX, featuresPerCell]);
    
    % Pattern combination method
    patternDistinctiveness = zeros(numCellsY, numCellsX);
    for i = 1:numCellsY
        for j = 1:numCellsX
            cellHistogram = double(squeeze(lbpPatterns(i,j,:)));
            normalizedHist = cellHistogram/sum(cellHistogram);
            patternDistinctiveness(i,j) = std(cellHistogram) * sum(-normalizedHist .* log2(normalizedHist + eps));
        end
    end
    
    % Apply weights and normalize
    patternVisualization = patternDistinctiveness .* weights;
    patternVisualization = (patternVisualization - min(patternVisualization(:))) / ...
                          (max(patternVisualization(:)) - min(patternVisualization(:)));
    
    imagesc(patternVisualization);
    colormap(gray);
    axis equal;
    axis tight;
    title('Weighted LBP Pattern Distribution');
    colorbar;
    
    % Add grid
    hold on;
    for i = 1:numCellsX-1
        line([i+0.5 i+0.5], [0.5 numCellsY+0.5], 'Color', 'w', 'LineWidth', 1);
    end
    for i = 1:numCellsY-1
        line([0.5 numCellsX+0.5], [i+0.5 i+0.5], 'Color', 'w', 'LineWidth', 1);
    end
end