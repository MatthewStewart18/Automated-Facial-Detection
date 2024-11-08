function fig = generate3DPcaPlot(pca_features, labels)
% Extract only the first 3 principal components for plotting
features_3D = pca_features(:, 1:3);

% Initialize 3D plot
figure;
hold on;
grid on;

% Define colors for each class (-1 for non-face, 1 for face)
colours = ['r', 'b'];

for i = [-1, 1]  % Non-face is -1, face is 1
    indexes = find(labels == i);
    fig = scatter3(features_3D(indexes, 1), features_3D(indexes, 2), features_3D(indexes, 3), ...
        30, colours((i + 3) / 2), 'filled'); % Adjust marker size and color here
end

% Label axes and add legend
xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');
legend('Non-face', 'Face');
title('3D Plot of First 3 Principal Components');
hold off;
end

