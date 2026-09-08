function [imagesCellTrain, meanAll, imageSize] = adjustFormat(imagesCellTrain)

% %compute image size
% %image size must be a power of 2
% im = imagesCellTrain{1};
% closPow2 = pow2(floor(log2(size(im,1))));
% imageSize = [closPow2 closPow2];

meanAll = zeros(numel(imagesCellTrain), 1);

for i = 1 : numel(imagesCellTrain)
    
    %load
    im = imagesCellTrain{i};
    
    %cast
    %im = double(im);
    
    %mean
    meanAll(i) = mean2(im);
    
    %subtract mean
    im = im - meanAll(i);
    
    %assign
    imagesCellTrain{i} = im;
    
end %for i = 1 : numel(imagesCellTrain)