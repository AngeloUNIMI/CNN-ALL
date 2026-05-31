function thFocusFinal = find_th_focus(imagesCellTrain, TrnLabels, imageSize, dirUtilities, fidLogs)


%read
focusAll_Trn = -1 .* ones(numel(imagesCellTrain), 1);
%loop read
parfor f = 1 : numel(imagesCellTrain)
% for f = 1 : numel(imagesCellTrain)
    
    %im
    im = imagesCellTrain{f};
    
    %resize
    im = imresize(im, imageSize);
    
    %focus
    [scoreFocus] = assessFocusFQPath(im, [], dirUtilities, 0);
    
    %assign
    focusAll_Trn(f) = scoreFocus;
    
end %for f


%correlation initial
correlationVanilla = computeCorrelation(focusAll_Trn', TrnLabels);
%display
%fprintf_pers(fidLogs, ['\tCorrelation initial: ' num2str(correlationVanilla) '\n']);


%--------------------------------------
%minimize correlation between focus and labels
%display
%fprintf_pers(fidLogs, '\tMinimize correlation\n');
%init
thFocus = 9;
focusProc = -1 .* ones(numel(imagesCellTrain), 1);
correlationProc = abs(correlationVanilla * 2);
correlationProcPrev = correlationProc * 2;
thFocusFinal = 1000;
correlationProcMin = 1000;

%loop on decreasing values of thFocus
%while abs(correlationProc) < abs(correlationProcPrev)
for ww = 1 : 30
%%%%%%%%%%%%%%%%%
% for ww = 1 : 5
%%%%%%%%%%%%%%%%%
    
    %display
    %fprintf_pers(fidLogs, ['\t\tthFocus: ' num2str(thFocus) '\n']);
    
    %loop
    parfor f = 1 : numel(imagesCellTrain)
        %for f = 1 : numel(imagesCellTrain)
        
        im = imagesCellTrain{f};
        
        %resize
        im = imresize(im, imageSize);
        
        %display
        %if mod(f, stepPrint) == 0
        %fprintf_pers(fidLogs, ['\t\t\t' filenameTrn{f} '\n']);
        %end %if mod
        
        %process
        radius = 1;
        iterF = 1;
        im_temp = im;
        [scoreFocus] = assessFocusFQPath(im, [], dirUtilities, 0);
        while scoreFocus > thFocus && iterF < 10
            im_temp = imsharpen(im, 'Radius', radius);
            [scoreFocus] = assessFocusFQPath(im_temp, [], dirUtilities, 0);
            radius = radius + 2;
            iterF = iterF + 1;
        end %for f
        im = im_temp;
        
        focusProc(f) = scoreFocus;
        
    end %for g
    
    %update correlation values
    correlationProcPrev = correlationProc;
    correlationProc = computeCorrelation(focusProc', TrnLabels);
    
    %
    if abs(correlationProc) < correlationProcMin
        correlationProcMin = abs(correlationProc);
        thFocusFinal = thFocus;
    end %if correlation
    
    %display
    %fprintf_pers(fidLogs, ['\t\t\tCorrelation: ' num2str(correlationProc) '\n']);
    
    %decrease thFocus
    thFocus = thFocus - 0.1;
    
end %while correlationProc


%display
fprintf_pers(fidLogs, ['\t\tth_focus: ' num2str(thFocusFinal) '\n']);



