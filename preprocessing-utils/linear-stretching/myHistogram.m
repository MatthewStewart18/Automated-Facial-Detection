function histogram = myHistogram(image)
    assert(isa(image, 'uint8'), 'Input must be uint8');

    % Initialize the histogram
    histogram = zeros(1, 256); % 256 bins for pixel values 0 to 255

    % Calculate histogram
    for value = 0:255
        histogram(value + 1) = sum(image(:) == value);
    end
end