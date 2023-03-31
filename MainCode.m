
clc
clear all

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
labelCount = countEachLabel(imds)
img = readimage(imds,2);
size(img)
numTrainFiles = 600;
numValFiles = 200;
[imdsTrain,imdsValidation,imdsVTest] = splitEachLabel(imds,numTrainFiles,numValFiles,'randomize');
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',6, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracyVal = sum(YPred == YValidation)/numel(YValidation)


%%  Test Accs
YPred = classify(net,imdsVTest);
YTest = imdsVTest.Labels;
accuracyTest= sum(YPred == YTest)/numel(YTest)


%%  test labelling
C = confusionmat(YPred,YTest); % gercek ve tahminler in sayisal olarak dogrulugu confusionMAtrixle tespit edilmistir.
figure
cm=confusionchart(YPred,YTest);
title('Testing Confusion Matrics ')