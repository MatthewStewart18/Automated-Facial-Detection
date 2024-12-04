clear all;
close all;

% Add paths
addpath ../../images
addpath ../../utils
addpath ../../feature-extraction-utils
addpath ../../feature-extraction-utils/feature-extractors
addpath ../../preprocessing-utils

% Load data
[Xtrain, Ytrain] = loadFaceImages('../../images/face_train.cdataset');
[Xtest, Ytest] = loadFaceImages('../../images/face_test.cdataset', -1);

% Normalize and convert data
Xtrain = double(Xtrain) / 255;
Xtest = double(Xtest) / 255;
Ytrain(Ytrain == -1) = 0; % Convert labels to binary
Ytest(Ytest == -1) = 0;

% Define and configure the neural network
net = patternnet(10);  % Adjust number of neurons if needed
net.trainFcn = 'trainlm';       % Levenberg-Marquardt backpropagation
net.performFcn = 'crossentropy'; % Cross-entropy loss
net.layers{end}.transferFcn = 'logsig'; % Sigmoid activation in output layer

% Divide data
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Train the network
[net, tr] = train(net, Xtrain', Ytrain');

% Predict probabilities on the test set
Yprobs = net(Xtest');

% Convert to binary predictions (threshold at 0.5)
Ypred = round(Yprobs);

% Calculate metrics
[accuracy, precision, recall, f1_score, confusionMatrix] = calculateMetrics(Ytest', Ypred);

% Plot ROC curve
figure;
[Xroc, Yroc, T, AUC] = perfcurve(Ytest, Yprobs, 1);
plot(Xroc, Yroc);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('Neural Net ROC Curve (AUC = %.2f)', AUC));

% Save the network
save('trainedFaceNet.mat', 'net');