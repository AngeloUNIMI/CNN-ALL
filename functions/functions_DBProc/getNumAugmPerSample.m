function [maxAugm] = getNumAugmPerSample(files)

%init
%maxAugm = -1;
augmNumV = zeros(numel(files), 1);
%loop on files
for i = 1 : numel(files)
    
    filename = files(i).name;
    
    %sample
    [C, ~] = strsplit(filename, {'_', '.'});
    sampleInd = str2double(C{1});
    
    %augm num
    %[C, indS] = strsplit(filename, {'_augm_', '.'}); 
    %augmNum = str2double(C{2});
    
    augmNumV(sampleInd) = augmNumV(sampleInd) + 1;
    
    %look for max augmNum
    %if augmNum > maxAugm
        %maxAugm = augmNum;
    %end %if augmNum
    
end %for i

maxAugm = max(augmNumV);

