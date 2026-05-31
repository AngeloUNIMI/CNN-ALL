
%param deblur
params.do_export = false; % export restored images
params.scale = 1/4;    % define the down-sampling scale (default is 1/4)
params.model_type = 'Gaussian';   % PSF model: 'Gaussian' or 'Laplacian'
params.significancy = 0.5; % edge significany control (optional) default is 0.5

%param focus
params.thFocus = 8;

%param segm
params.offsetOtsuIntens = 0;
params.numberMultiThresh = 2;
params.typeSeSegm = 'disk'; 
params.sizeSeSegm = 5;
params.numIterAC = 500;

%params clustering
params.nClusters = 3;
params.classesToConsider = 2;
params.offsetOtsuFCM = 0.2;

%params extract roi
params.axisScaleFac = 1.5;
params.roiSize = [256 256];



