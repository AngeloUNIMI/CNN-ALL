function [H, E, Bg] = deConvStain(im, filename, dirResults, plotta, savefile)

%Alsubaie N, Trahearn N, Raza SEA, Snead D, Rajpoot NM (2017)
%Stain Deconvolution Using Statistical Analysis of Multi-Resolution Stain Colour Representation.
%PLOS ONE 12(1): e0169875.

[~, ~, H, E, Bg] = SCD_MA(im, plotta);

offsetHist = 50;

% figure, imshow(H)
% figure, imshow(E)
% figure, imshow(Bg)


%find E

%the one with greatest red channel
[cH, xH] = imhist(H(:,:,1));
[cE, xE] = imhist(E(:,:,1));
[cBg, xBg] = imhist(Bg(:,:,1));
sumH = sum(cH(end-offsetHist : end));
sumE = sum(cE(end-offsetHist : end));
sumBg = sum(cBg(end-offsetHist : end));
[maxSum, indM] = max([sumH, sumE, sumBg]);

%cell
stainCell{1} = H;
stainCell{2} = E;
stainCell{3} = Bg;

%switch
appoggio = stainCell{2};
stainCell{2} = stainCell{indM};
stainCell{indM} = appoggio;

%revert
if 1
    H = stainCell{1};
    E = stainCell{2};
    Bg = stainCell{3};
end %if 0



if plotta
    fh = fsfigure;
    
    subplot(1, 4, 1);
    imshow(im);
    title('Source');
    
    subplot(1, 4, 2),
    imshow(H);
    title('H');
    
    subplot(1, 4, 3),
    imshow(E);
    title('E');
    
    subplot(1, 4, 4),
    imshow(Bg);
    title('Bg');

    str = [filename '; Stain deconvolution'];
    
    mtit(fh, str, 'Interpreter', 'none', 'fontsize', 14, 'color', [1 0 0], 'xoff', .0, 'yoff', .0);
    
    %save
    if savefile
        export_fig(gcf, [dirResults str '.jpg']);
    end %if save
    
end %if plotta




