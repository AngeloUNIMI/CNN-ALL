clc
close all
clear variables
fclose('all');
addpath(genpath('./steps'));
addpath(genpath('./libraries'));
addpath(genpath('./functions'));
addpath(genpath('./util'));
dirUtilities = './libraries/lib_FQPath/utilities/';

%General parameters
savefile = 1;
logS = 1;
fidLogs{1} = 1; %stdoutput
%multi-core
numCoresFeatExtr = 8;
numCoresKnn = 2;
%ext
ext = 'tif';
%dirs
dirOrig = '.\imgs\orig\';
dirTest = '.\imgs\test\';
mkdir_pers(dirTest, savefile);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%A. process DB files
fprintf(1, 'Processing DB...\n\n\n');
step_A_process_ALL_IDB(dirOrig, dirTest, ext, fidLogs, savefile);
clc
close all
pause(1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%B. registration
plotFigures = 0;
fprintf(1, 'Registering images...\n\n\n');
step_B_registration(dirTest, dirUtilities, ext, fidLogs, logS, savefile, plotFigures);
clc
close all
pause(1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%C. VAR-PCANet
plotFigures = 1;
fprintf(1, 'Adaptive unsharpening using VAR-PCANet...\n\n\n');
step_C_varpcanet(dirTest, dirUtilities, ext, numCoresFeatExtr, numCoresKnn, fidLogs, logS, savefile, plotFigures);




























