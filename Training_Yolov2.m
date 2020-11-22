
[imds,bxds] = objectDetectorTrainingData(gTruth);

ds = combine(imds, bxds);
% Create YOLOv2 Object Detection Network
inputSize = [224 224 3];
numClasses = 4;
augmentTrainingData = transform(ds, @augmentData);
trainingDataForEstimation = transform(augmentTrainingData,@(data)preprocessData(data,inputSize));
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);
featureExtraction = resnet50;
featureLayer = 'activation_40_relu';
lgraph = yolov2Layers(inputSize, numClasses, anchorBoxes, featureExtraction, featureLayer);
% Training
options = trainingOptions('sgdm',...
          'InitialLearnRate',0.001,...
          'Verbose',true,...
          'MiniBatchSize',20,...
          'MaxEpochs',30,...
          'Shuffle','never',...
          'VerboseFrequency',10,...
          'CheckpointPath',tempdir);
 
[detector, info] = trainYOLOv2ObjectDetector(trainingDataForEstimation, lgraph, options);
save('detector_augment_Yolov2_150.mat', 'detector');
save('info_Yolov2__150.mat', 'info');

