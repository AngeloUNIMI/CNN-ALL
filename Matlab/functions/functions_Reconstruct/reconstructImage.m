function [tmpImage, errV] = reconstructImage(resF, sortRes, gaborBank, powerMapF, indFiltersToUse, im, imageSize, sizeRes, param, plotta)

%init matrix indicating if for each position a wavelet
%with orient and scale already used
tmpRes = initTmpRes(gaborBank, powerMapF);

%init error vector
errV = zeros(param.numWavelets, 1);

%init reconstructed image
tmpImage = zeros(imageSize);

%create figure
if plotta
    fsfigure
end %if plotta

%init counters
countW = 1; %wavelet counter
ind = 1; %sorted response counter
%loop on wavelet responses
while (countW <= param.numWavelets && ind <= sizeRes)
    
    %get information of corresponding wavelet
    currScale = sortRes(ind, 2);
    currTheta = sortRes(ind, 3);
    currX = sortRes(ind, 4);
    currY = sortRes(ind, 5);
    currO = sortRes(ind, 6);
    
    %check if current wavelet at current position already
    %used
    if (tmpRes(currO).value(currY, currX) == 1)
        %increment counter
        ind = ind + 1;
        continue
    end %if (tmpRes(currY, currX, currO)
    
    %check if current wavelet among the best
    if ~ismember(currO,  indFiltersToUse)
        %increment counter and skip
        ind = ind + 1;
        continue
    end %if ~ismember(currO,  ind_o_counter_sort)
    
    %get filters with current parameters
    currGaborEven = gaborBank(currO).even;
    currGaborOdd = gaborBank(currO).odd;
    
    %get weights for gabor filters
    weightEven = resF(currO).even;
    weightOdd = resF(currO).odd;
    
    %compute filtersize
    filterSize = computeFilterSizeFromScale(currScale);
    %compute step
    step = computeStepFromScale(currScale);
    
    %get filter position
    %check x,y correct
    [posX, posY, posX_filter, posY_filter] = findFilterPosition(currX, currY, step, imageSize(1), filterSize);
    
    %shift filter to center filter at x,y
    %currGaborEven = adjustGaborPos(currGaborEven, currX, currY, imageSize);
    %currGaborOdd = adjustGaborPos(currGaborOdd, currX, currY, imageSize);
    
    %add to image
    %tmpImage = tmpImage + (weightEven(currY, currX) * currGaborEven);
    %tmpImage = tmpImage + (weightOdd(currY, currX) * currGaborOdd);
    
    %add to image
    tmpImage(posY, posX) = tmpImage(posY, posX) + (weightEven(currY, currX) * currGaborEven(posY_filter, posX_filter));
    tmpImage(posY, posX) = tmpImage(posY, posX) + (weightOdd(currY, currX) * currGaborOdd(posY_filter, posX_filter));
    
    %Mean Squared Error between original image and
    %reconstructed
    imOrigNorm = normalizzaImg(im);
    err = immse(imOrigNorm, normalizzaImg(tmpImage));
    errV(countW) = err;
        
    if plotta && mod(countW, 1000) == 0
        
        %get image limits
        climG = getImageLimits(currGaborEven);
        climI = getImageLimits(tmpImage);
        
        
        subplot(2,2,1);
        imshow(im,[])
        title('Original Image')
        
        subplot(2,2,2);
        imshow(currGaborEven, climG);
        title('Current wavelet')
        
        subplot(2,2,3);
        imshow(tmpImage, climI)
        title({['Reconstructed image with ' num2str(numel(indFiltersToUse)) ' filters, after ' num2str(countW) ' wavelets'], ['MSE: ' num2str(err)]});
        
        subplot(2,2,4);
        plot(errV(1:10:countW), 'LineWidth', 2);
        xlabel('Number of wavelets added')
        ylabel('MSE')
        title('MSE of reconstruction')
        
        pause(0.01)
        
    end %if mod(countW, 100) == 0
    
    %increment counters
    countW = countW + 1;
    ind = ind + 1;
    
    %threshold
    %if (err < param.minMSE) && (countW > param.minCountW)
    %break
    %end %if err < param.minMSE
    
end %while (countW <= param.numWavelets && ind <= sizeRes)

