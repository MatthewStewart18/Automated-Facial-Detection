function imageList = reshape2dImage(trainingImages)
    for image = 1:size(trainingImages, 1)
        imagePixelVector = trainingImages(image,:);
        imageMatrix = reshape(imagePixelVector, [27, 18]);
        imageList{image} = imageMatrix;
    end


