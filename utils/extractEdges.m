function [edges] = extractEdges(images)
edges = zeros(size(images, 1), 27*18);
for i = 1:size(images, 1)
    img = reshape(images(i,:), [27, 18]);
    % Extract edges using Canny
    current_edge = edge(img,'Canny');
    current_edge = current_edge(:)';
    edges(i, :) = double(current_edge);
end
end

