function [] = step_B_registration(dirTest, dirUtilities, ext, fidLogs, logS, savefile, plotFigures)

%dbname
dbname = 'ALL_IDB';

%part
dbPart = 'ALL_IDB2';

%params
run('./params/params_registration.m');

%dirIn e Out
dirIn = [dirTest dbname '\' dbPart '\'];
dirOut = [dirIn 'ROI_' num2str(params.roiSize(1)) '/'];
dirOutColorNorm = [dirIn 'colorNorm\'];
mkdir_pers(dirOut, savefile);
mkdir_pers(dirOutColorNorm, savefile);

%dirResults
dirResults = './Results/';
mkdir_pers(dirResults, savefile);
fileSaveTest = [dirResults 'save.mat'];
%RESULTS: log file
timeStampRaw = datestr(datetime);
timeStamp = strrep(timeStampRaw, ':', '-');
if savefile && logS
    logFile = [dirResults dbname '_log_' timeStamp '.txt'];
    fidLog = fopen(logFile, 'w');
    fidLogs{2} = fidLog;
end %if savefile && log

%loop
files = dir([dirIn '*.' ext]);

for i = 1 : numel(files)
% for i = [1 2 3 141 142 143]
    
    %read im
    filename = files(i).name;
    im = imread([dirIn filename]);
    
    %display
    fprintf_pers(fidLogs, ['Im: ' filename]);
    
    
    %super-resolution?
    %illumination compensation?
    
    
    %----------------------------------------------------------------------
    %preprocessing
    
    %deblur - 1shot maxpol
    %im_deBlurred = deBlur1shotMaxPol(im, params);
    
    %focus assessment
    focusOriginal = assessFocusFQPath(im, filename, dirUtilities, plotFigures);
    fprintf_pers(fidLogs, ['\tFocus: ' num2str(focusOriginal)]);
    
    %skip de-focused images?
    if focusOriginal > params.thFocus
        fprintf_pers(fidLogs, '\tFocus low...');
        %continue
    end %if focus
    
    %color adjust
    imColorNorm = imNormalization(im, filename, plotFigures);
    %save+
    if savefile
        imwrite(imColorNorm, [dirOutColorNorm filename]);
    end %if save
    
    
    %----------------------------------------------------------------------
    %stain separation
    [H, E, Bg] = deConvStain(im, filename, dirResults, plotFigures, savefile);
    
    
    %----------------------------------------------------------------------
    %segmentation
    mask = segmentStain(im, imColorNorm, filename, dirResults, params, plotFigures, savefile);
    
    
    %----------------------------------------------------------------------
    %extraction of ROI
    [ROI, errorV] = extractROI(im, filename, mask, dirResults, params, plotFigures, savefile);
    if errorV == -1
        fprintf_pers(fidLogs, '\tCannot extract ROI...\n');
        continue
    else %if errorV
        fprintf_pers(fidLogs, '\tROI extracted');
    end %if errorV
    
    %write
    if savefile
        imwrite(ROI, [dirOut filename]);
    end %if save
    
    
    
    %----------------------------------------------------------------------
    %structure
    problem(i).filename = filename;
    %class
    [C, ~] = strsplit(filename, {'_', '.'});
    problem(i).class = str2double(C{2});
    %focus
    problem(i).focus = focusOriginal;
    

    %----------------------------------------------------------------------
    %pause
    if plotFigures
        pause(1)
        close all
        pause(0.1)
    end %if plotta
    
    
    %newline
    fprintf_pers(fidLogs, '\n');
    
end %for i


%----------------------------------------------------------------------
%save
save([dirOut 'classes.mat'], 'problem');


