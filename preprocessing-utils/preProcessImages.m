function [train, test] = preProcessImages(training_images, test_images)

train = zeros(size(training_images, 1), 27*18);
for i = 1:size(training_images, 1)
    img = reshape(training_images(i,:), [27, 18]);
    img = histeq(uint8(img));
    img = img(:)';
    train(i, :) = double(img);
end

test = zeros(size(test_images, 1), 27*18);
for i = 1:size(test_images, 1)
    img = reshape(test_images(i,:), [27, 18]);
    img = histeq(uint8(img));
    img = img(:)';
    train(i, :) = double(img);
end

end

