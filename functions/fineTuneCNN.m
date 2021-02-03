function [netTransfer] = fineTuneCNN(imagesCell, Labels, folder, inputSize, imageAugmenter, layers, options)

% imdsTrain = imageDatastore(folder, 'IncludeSubfolders', true, 'LabelSource','foldernames');

im_temp = imagesCell{1};
imsizeOrig = size(im_temp);
imArray = zeros(imsizeOrig(1), imsizeOrig(2), imsizeOrig(3), numel(imagesCell));
for ind_im = 1 : numel(imagesCell)
    imArray(:,:,:,ind_im) = imagesCell{ind_im};
end %for ind_im

% whos imArray Labels

% augimdsTrainl = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', imageAugmenter);
% augimdsTrainl = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augimdsTrainl = augmentedImageDatastore(inputSize(1:2), imArray, categorical(Labels)', 'DataAugmentation', imageAugmenter);

netTransfer = trainNetwork(augimdsTrainl, layers, options);