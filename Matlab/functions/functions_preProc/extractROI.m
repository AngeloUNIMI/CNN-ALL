function [ROI_rz, errorV] = extractROI(im, filename, mask, dirResults, params, plotta, savefile)

%init
errorV = 0;

%ellipse fit
[ix, jy] = find(edge(mask, 'sobel'));
ellipse_t = fit_ellipse(ix, jy, im, filename, mask, dirResults, plotta, savefile);
minAxis = ellipse_t.short_axis;

%center
%center = [ellipse_t.X0_in, ellipse_t.Y0_in];
%center = [mean(ix), mean(jy)];
%in ALL-IDB2, center of the image
center = [floor(size(im,1)/2), floor(size(im,2)/2)];

%select
radiusSelect = round(minAxis/2 * params.axisScaleFac);
centerRound = round(center);
try %maybe goes beyond image
    ROI = im(centerRound(2) - radiusSelect : centerRound(2) + radiusSelect, centerRound(1) - radiusSelect : centerRound(1) + radiusSelect, :);
catch %try
    try %try
        radiusSelect = round(minAxis/2);
        ROI = im(centerRound(2) - radiusSelect : centerRound(2) + radiusSelect, centerRound(1) - radiusSelect : centerRound(1) + radiusSelect, :);
    catch %try
        ROI_rz = [];
        errorV = -1;
        return;
    end %try
end %try

%resize
ROI_rz = imresize(ROI, params.roiSize);

%display
if plotta
    fh = figure;
    subplot(1,2,1)
    imshow(im)
    title('Original')
    subplot(1,2,2)
    imshow(ROI)
    title('ROI');
    
    str = [filename '; ROI extraction'];
    
    mtit(fh, [filename '; ROI extraction'], 'Interpreter', 'none', 'fontsize', 14, 'color', [1 0 0], 'xoff', .0, 'yoff', .0);
    
    if savefile
        export_fig(gcf, [dirResults str '.jpg']);
    end %if save
    
end %if plotta



