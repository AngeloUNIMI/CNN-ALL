function [imagesCellTrain, filenameTrn, indexes_imUnsharpened] = loadImages(files, dirDB, allIndexes, indImagesTrain, numImagesTrain, param, colorS, th_focus, dirUtilities, plotta)

%init
vectorIndexTrain = allIndexes(indImagesTrain);
filenameTrn = cell(length(vectorIndexTrain), 1);
imagesCellTrain = cell(numImagesTrain, 1);

%init maxs
maxH = -1;
maxW = -1;
%loop - find max
for i = 1 : length(vectorIndexTrain)
    
    %read
    filenameTrn{i} = files(vectorIndexTrain(i)).name;
    %im = im2double(imread([dirDB filenameTrn{i}]));
    im = imread([dirDB filenameTrn{i}]);
    
    [h, w, ~] = size(im);
    if h > maxH
        maxH = h;
    end %if h
    if w > maxW
        maxW = w;
    end %if w
    
end %for i


%read images
parfor i = 1 : length(vectorIndexTrain)
% for i = 1 : length(vectorIndexTrain)
    
    %read
    filenameTrn{i} = files(vectorIndexTrain(i)).name;
    %im = im2double(imread([dirDB filenameTrn{i}]));
    im = imread([dirDB filenameTrn{i}]);
      
    if colorS
        imagesCellTrain{i, 1} = im;
    else %if colorS
        imagesCellTrain{i, 1} = rgb2gray(im);
    end %if colorS
    
end %for i = 1 : numImages


%resize to largest dimension
parfor i = 1 : numel(imagesCellTrain)
    im =  imagesCellTrain{i, 1};
    
    im = imresize(im, [maxH, maxW]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %im = imresize(im, [maxH/2, maxW/2]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %im = imresize(im, [maxH/4, maxW/4]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    
    imagesCellTrain{i, 1} = im;
    
end %for i

%preprocess
indexes_imUnsharpened = false(length(vectorIndexTrain));
parfor i = 1 : length(vectorIndexTrain)
% for i = 1 : length(vectorIndexTrain)
    
    %read
    im = imagesCellTrain{i, 1};
    
    %process
    f = 1;
    iterF = 1;
    im_temp = im;
    [scoreFocus] = assessFocusFQPath(im, [], dirUtilities, 0);
    while scoreFocus > th_focus && iterF < 10
        im_temp = imsharpen(im, 'Radius', f);
        [scoreFocus] = assessFocusFQPath(im_temp, [], dirUtilities, 0);
        f = f + 2;
        iterF = iterF + 1;
        
        indexes_imUnsharpened(i) = true;
    end %for f
    
    %assign
    imagesCellTrain{i, 1} = im_temp;   
    
end %for i = 1 : numImages


%display
if plotta
    numSample = min([144, numel(imagesCellTrain)]);
    indexM = randsample(numel(imagesCellTrain), numSample);
    figure,
    montage(imagesCellTrain(indexM), 'DisplayRange', [], 'Size', [9 16])
    title('Random subset of images');   
end %if plotta

