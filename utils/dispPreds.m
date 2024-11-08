function dispPreds(predictions, test_labels, test_images)
% Display 100 correctly classified images
comparison = (test_labels == predictions);
figure;
sgtitle('Correct Classifications');

count = 0;
i = 1;
while (count < 100) && (i <= length(comparison))
    if comparison(i)
        count = count + 1;
        subplot(10, 10, count);
        img = reshape(test_images(i, :), [27, 18]);
        imshow(uint8(img));
        title(sprintf('Label: %d', test_labels(i))); % Display correct label
    end
    i = i + 1;
end

% Display all incorrectly classified images
figure;
incorrect_indices = find(~comparison);
num_incorrect = length(incorrect_indices);
grid_size = ceil(sqrt(num_incorrect)); 
sgtitle('Incorrect Classifications');

for j = 1:num_incorrect
    subplot(grid_size, grid_size, j);
    idx = incorrect_indices(j);
    img = reshape(test_images(idx, :), [27, 18]);
    imshow(uint8(img));
    title(sprintf('Pred: %d, Actual: %d', predictions(idx), test_labels(idx))); % Display predicted and true labels
end
end