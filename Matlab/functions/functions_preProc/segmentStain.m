function [maskACClopeOpen] = segmentStain(im, imColorNorm, filename, dirResults, params, plotta, savefile)

%------------------------------------
%otsu (intensity) on E
if size(imColorNorm, 3) == 3
    imColorNorm_gray = rgb2gray(imColorNorm);
end %if size

th = graythresh(imColorNorm_gray) + params.offsetOtsuIntens;

%thresh = imColorNorm_gray < th*255;
thresh = imColorNorm_gray < th;
threshOne = connComp2(thresh, 1, 1);

maskThresh = threshOne;

% figure
% imshow(maskThresh)
% pause


%------------------------------------
%fuzzy c-means on E
%imAlt = im2uint8(rgb2lab(im));
imAlt = im2uint8(rgb2hsv(im));

[C, U, LUT, H] = FastFCMeans(imAlt, params.nClusters);
%analyze centers, discard greatest one (white background)
% [C_sort, indC] = sort(C, 'ascend');
[C_sort, indC] = sort(C, 'descend'); %for hsv
C_select = indC(1 : params.classesToConsider);  
% %memberships to maps
L = LUT2label(imAlt, LUT);
Umap = FM2map(imAlt, U, H);

%memberships
if plotta
    fh1 = figure;
    subplot(1,3,1)
    imshow(rgb2gray(Umap(:,:,:,1)),[]);
    subplot(1,3,2)
    imshow(rgb2gray(Umap(:,:,:,2)),[]);
    subplot(1,3,3)
    imshow(rgb2gray(Umap(:,:,:,3)),[]);
    
    str = [filename '; class memberships'];
    
    mtit(fh1, str, 'Interpreter', 'none', 'fontsize', 14, 'color', [1 0 0], 'xoff', .0, 'yoff', .0);
    
    if savefile
        export_fig(gcf, [dirResults str '.jpg']);
    end %if save
    
end %if plotta

%[C, U, LUT, H] = FastFCMeans(im, params.nClusters);
%analyze centers, discard greatest one (white background)
% [C_sort, indC] = sort(C, 'ascend');
% C_select = indC(1 : params.classesToConsider);
% %memberships to maps
% L = LUT2label(im, LUT);
% Umap = FM2map(im, U, H);


%select
maskFuzzy = false([size(L, 1) size(L,2)]);
classThresholded = cell(numel(C_select), 1);
for c = 1 : numel(C_select)
    class = rgb2gray(Umap(:,:,:,C_select(c)));
    
    class = imadjust(class);
    
    %for one of the classes, modify the threshold
    %if c == 2
        threshClass = graythresh(class) + params.offsetOtsuFCM;
    %else %if c
        %threshClass = graythresh(class);
    %end %if c
    
    classThresholded{c} = class > threshClass;
    
    %threshClass = multithresh(class, params.numberMultiThresh) - params.offsetOtsuFCM;
    %classThresholded{c} = class > threshClass(2);
    
    maskFuzzy = logical(maskFuzzy + classThresholded{c});
end %for c
maskFuzzy = connComp2(maskFuzzy, 1, 1);

%display
if plotta
    fh2 = figure;
    for c = 1 : numel(C_select)
        subplot(1, numel(C_select), c)
        imshow(classThresholded{c})
    end %for c
    
    str = [filename '; class memberships, segmentation'];
    
    mtit(fh2, str, 'Interpreter', 'none', 'fontsize', 14, 'color', [1 0 0], 'xoff', .0, 'yoff', .0);
    
    if savefile
        export_fig(gcf, [dirResults str '.jpg']);
    end %if save
    
end %if plotta


%------------------------------------
%combine
% mask = logical(maskThresh .* maskFuzzy);
mask = logical(maskThresh + maskFuzzy);

%morph
se = strel(params.typeSeSegm, params.sizeSeSegm);
maskFill = imfill(mask, 'holes');
maskClose = imclose(maskFill, se);
maskCloseOpen = imopen(maskClose, se);


%------------------------------------
%refine with active contour
maskAC = activecontour(im, maskCloseOpen, params.numIterAC, 'Chan-Vese', 'ContractionBias', -0.3);
%morph
maskAC_oneCC = connComp2(maskAC, 1, 1);
maskACFill = imfill(maskAC_oneCC, 'holes');
maskACClose = imclose(maskACFill, se);
maskACClopeOpen = imopen(maskACClose, se);



%------------------------------------
if plotta
    fh = fsfigure;
    
    subplot(2,3,1)
    imshow(im)
    title('Original')
    
    subplot(2,3,2)
    imshow(im2double(im) + edge(maskThresh), []);
    title('Mask Otsu');
    
    subplot(2,3,3)
    imshow(im2double(im) + edge(maskFuzzy), []);
    title('Mask Fuzzy C-Means');
    
    subplot(2,3,4)
    imshow(im2double(im) + edge(maskCloseOpen), []);
    title('Mask combined + morph');
    
    subplot(2,3,5)
    imshow(im2double(im) + edge(maskAC), []);
    title('Mask combined + morph + AC');
    
    subplot(2,3,6)
    imshow(im2double(im) + edge(maskACClopeOpen), []);
    title('Mask combined + morph + AC + morph');
    
    str = [filename '; segmentation'];
    
    mtit(fh, str, 'Interpreter', 'none', 'fontsize', 14, 'color', [1 0 0], 'xoff', .0, 'yoff', .04);
    
    if savefile
        export_fig(gcf, [dirResults str '.jpg']);
    end %if save
    
end %if plotta




