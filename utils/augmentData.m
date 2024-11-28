function [aug_images, aug_labels] = augmentData(images, labels, image_size)
    % Inputs:
    % - images: Matrix of flattened grayscale images (rows = images)
    % - labels: Vector of corresponding labels
    % - image_size: [height, width] of the original images
    %
    % Outputs:
    % - aug_images: Matrix of augmented images
    % - aug_labels: Vector of augmented labels

    % Initialize augmented images and labels
    aug_images = [];
    aug_labels = [];
    
    % Loop through each image
    for i = 1:size(images, 1)
        % Get the current image and label
        label = labels(i);
        image_flat = images(i, :);
        
        % Reshape to original size
        image = reshape(image_flat, image_size);
        
        % Add the original image to augmented set
        aug_images = [aug_images; image_flat];
        aug_labels = [aug_labels; label];
        
        % Apply augmentations
        % Flip and shift operations for label 1 (e.g., face images)
        if label == 1
            transformations = {
                @fliplr, ...
                @(I) circshift(I, 1), ...
                @(I) circshift(I, -1), ...
                @(I) circshift(I, [0 1]), ...
                @(I) circshift(I, [0 -1]), ...
                @(I) circshift(fliplr(I), 1), ...
                @(I) circshift(fliplr(I), -1), ...
                @(I) circshift(fliplr(I), [0 1]), ...
                @(I) circshift(fliplr(I), [0 -1])
            };
        else
            % Simpler flips for non-face images
            transformations = {
                @fliplr, ...
                @flipud, ...
                @(I) flipud(fliplr(I))
            };
        end
        
        % Apply each transformation and add to augmented dataset
        for transform = transformations
            I_aug = transform{1}(image);
            vector_aug = reshape(I_aug, 1, []);
            aug_images = [aug_images; vector_aug];
            aug_labels = [aug_labels; label];
        end
    end
end
