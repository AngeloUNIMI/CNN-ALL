function [errorStruct] = computeClassPerformanceFineTuneCNN(imagesCell, Labels, folder, inputSize, netTransfer, fidLogs)

% imdsTest = imageDatastore(folder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

im_temp = imagesCell{1};
imsizeOrig = size(im_temp);
imArray = zeros(imsizeOrig(1), imsizeOrig(2), imsizeOrig(3), numel(imagesCell));
for ind_im = 1 : numel(imagesCell)
    imArray(:,:,:,ind_im) = imagesCell{ind_im};
end %for ind_im

% imdsTestAugm = augmentedImageDatastore(inputSize(1:2), imdsTest);
imdsTestAugm = augmentedImageDatastore(inputSize(1:2), imArray, categorical(Labels)');

tic
%[TestOutput, scores] = classify(netTransfer, imArrayTest);
[TestOutput, scores] = classify(netTransfer, imdsTestAugm, 'MiniBatchSize', 10);

% scores
% TestOutput
% double(imdsTest.Labels)
% oneHotLabels = onehot(double(imdsTest.Labels));
% correlationProc = computeCorrelation(scores, oneHotLabels)

%cast
%TestOutput = double(TestOutput)-1;
timeClass = toc;
fprintf_pers(fidLogs, ['\t\tTime for classification: ' num2str(timeClass) ' s\n']);
%Confusion matrix
%C_knn = confusionmat(TestLabels, TestOutput);
C_knn = confusionmat(categorical(Labels), TestOutput);

%Error metrics
errorStruct = computeErrorsFromCM(C_knn);

