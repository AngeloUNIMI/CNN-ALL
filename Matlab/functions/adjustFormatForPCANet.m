function [imagesCellTrain, imageSize] = adjustFormatForPCANet(imagesCellTrain, imageSize)

if nargin == 1
    %compute image size
    %image size must be a power of 2
    im = imagesCellTrain{1};
    closPow2 = pow2(floor(log2(size(im,1))));
    imageSize = [closPow2 closPow2];
end

for i = 1 : numel(imagesCellTrain)
    
    %load
    im = imagesCellTrain{i};
    
    im = im2double(im);
    
    if size(im, 3)
        im = rgb2gray(im);
    end %if size
    
    %image size must be a power of 2
    im = imresize(im, imageSize);
    
    %assign
    imagesCellTrain{i} = im;
    
end %for i = 1 : numel(imagesCellTrain)