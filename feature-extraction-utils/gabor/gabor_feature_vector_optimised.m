function [vector, feature_mask] = gabor_feature_vector_optimised(image, feature_mask)
    % Load pre-computed Gabor filters
    load gabor;
    
    % Apply adaptive histogram equalization with adjusted parameters
    image = adapthisteq(image, 'Numtiles', [8 3], 'ClipLimit', 0.02);
    
    % Initialize feature cell array
    Features135x144 = cell(5,8);
    
    % Calculate dimensions
    img_height = 27;
    img_width = 18;
    num_scales = 4;
    num_orientations = 7;
    
    % Use more scales and orientations (4 scales and 7 orientations)
    for s = 1:num_scales
        for j = 1:num_orientations
            Features135x144{s,j} = abs(ifft2(G{s,j}.*fft2(double(image),32,32),img_height,img_width));
        end
    end
    
    % Convert cell array to matrix
    Features45x48 = cell2mat(Features135x144);
    
    % Calculate total feature size
    total_features = img_height * img_width * num_scales * num_orientations;
    
    % Reshape to vector
    full_vector = reshape(Features45x48, [1 total_features]);
    
    % If feature mask is not provided, create one
    if nargin < 2
        % Create temporary feature matrix for variance analysis
        temp_features = zeros(10, length(full_vector));
        
        % Generate features for a few random shifts of the image
        for i = 1:10
            shifted_image = circshift(image, [randi([-2,2]) randi([-2,2])]);
            Features135x144_temp = cell(5,8);
            for s = 1:num_scales
                for j = 1:num_orientations
                    Features135x144_temp{s,j} = abs(ifft2(G{s,j}.*fft2(double(shifted_image),32,32),img_height,img_width));
                end
            end
            Features45x48_temp = cell2mat(Features135x144_temp);
            temp_features(i,:) = reshape(Features45x48_temp, [1 total_features]);
        end
        
        % Calculate variance of each feature across shifted versions
        feature_vars = var(temp_features);
        
        % Select top 2000 most variant features (adjusted number)
        [~, sorted_indices] = sort(feature_vars, 'descend');
        feature_mask = false(1, length(full_vector));
        feature_mask(sorted_indices(1:2000)) = true;
    end
    
    % Apply feature mask to get final vector
    vector = full_vector(feature_mask);
    
    % Normalize the feature vector with robust scaling
    vector = (vector - median(vector)) ./ (iqr(vector) + eps);
    
    % Debug information
    fprintf('Feature vector size: %d\n', length(vector));
end