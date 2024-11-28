% Function to visualize HOG features
function visualizeHOG(image, cellSize, blockSize)
    % Reshape the image
    im = reshape(image, [27, 18]);
    
    % Calculate gradients
    [Ix, Iy] = gradient(double(im));
    magnitude = sqrt(Ix.^2 + Iy.^2);
    angle = atan2d(Iy, Ix);
    angle = mod(angle, 180);  % Convert to 0-180 range
    
    % Create visualization
    figure('Name', 'HOG Visualization');
    
    % Subplot 1: Original Image
    subplot(2,2,1);
    imshow(im, []);
    title('Original Image');
    
    % Subplot 2: Gradient Magnitude
    subplot(2,2,2);
    imshow(magnitude, []);
    title('Gradient Magnitude');
    
    % Subplot 3: Gradient Directions
    subplot(2,2,3);
    imshow(angle, [0 180]);
    colormap(gca, 'jet');
    colorbar;
    title('Gradient Directions');
    
    % Subplot 4: HOG cells overlay
    subplot(2,2,4);
    imshow(im, []);
    hold on;
    
    % Draw cell grid
    for i = 1:cellSize:size(im,1)
        line([1,size(im,2)], [i,i], 'Color', 'r', 'LineStyle', ':');
    end
    for j = 1:cellSize:size(im,2)
        line([j,j], [1,size(im,1)], 'Color', 'r', 'LineStyle', ':');
    end
    
    % Draw block grid
    for i = 1:blockSize:size(im,1)
        line([1,size(im,2)], [i,i], 'Color', 'b');
    end
    for j = 1:blockSize:size(im,2)
        line([j,j], [1,size(im,1)], 'Color', 'b');
    end
    
    title('HOG Cell/Block Structure');
    % legend('Cells', 'Blocks');
    hold off;
end