function vector = gabor_feature_vector_subset(image)

    % Load Gabor filters
    load gabor;

    % Apply adaptive histogram equalization
    image = adapthisteq(image, 'Numtiles', [8, 3]);

    % Initialize cell array to store features
    sampledFeatures = cell(2, 4);  % Use only 2 scales and 4 orientations as an example

    % Sample subset of Gabor filters
    scales = [1, 3];       % Choose only scales 1 and 3
    orientations = [1, 3, 5, 7]; % Choose only orientations 1, 3, 5, and 7

    % Loop over selected scales and orientations
    index = 1;
    for s = scales
        for j = orientations
            % Compute Gabor feature and store in sampledFeatures
            sampledFeatures{index} = abs(ifft2(G{s, j} .* fft2(double(image), 32, 32), 27, 18));
            index = index + 1;
        end
    end

    % Convert cell array to matrix and reshape into a vector
    sampledFeaturesMatrix = cell2mat(sampledFeatures);
    vector = reshape(sampledFeaturesMatrix, [1, numel(sampledFeaturesMatrix)]);

end