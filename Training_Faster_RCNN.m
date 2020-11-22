
[imds,bxds] = objectDetectorTrainingData(gTruth);

ds = combine(imds, bxds);
% Create FasterRCNN Object Detection Network
inputSize = [224 224 3];
numClasses = 4;
augmentTrainingData = transform(ds, @augmentData);
trainingDataForEstimation = transform(augmentTrainingData,@(data)preprocessData(data,inputSize));
numAnchors = 3;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);
featureExtraction = resnet50;
featureLayer = 'activation_40_relu';
lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtraction,featureLayer);
% Training
options = trainingOptions('sgdm',...
          'InitialLearnRate',0.001,...
          'Verbose',true,...
          'MiniBatchSize',2,...
          'MaxEpochs',10,...
          'Shuffle','never',...
          'VerboseFrequency',10,...
          'CheckpointPath','');
[detector, info] = trainFasterRCNNObjectDetector(trainingDataForEstimation,lgraph,options);
save('detector_augment_150_FasterRCNN.mat', 'detector');
save('info_FasterRCNN_150.mat', 'info');

