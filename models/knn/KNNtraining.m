function modelKNN = KNNtraining(images, labels, K)
    modelKNN.neighbours=images;
    modelKNN.labels=labels;
    modelKNN.K=K;
end