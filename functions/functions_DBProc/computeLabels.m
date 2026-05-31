function [problem, labels, numImagesAll] = computeLabels(dirDB, files)


%load mat
filesMat = dir([dirDB '*.mat']);
mat = load([dirDB filesMat(1).name]);
problem = mat.problem;



%keep only elements which are in 'files'
indexrem = [];
for p = 1 : numel(problem)
    
    %get filename from problem
    filenameP = problem(p).filename;
    
    %init
    findP = -1;
    
    %loop on files
    for f = 1 : numel(files)
        
        %get filename from files
        filenameF = files(f).name;
        
        %cmp
        if strcmp(filenameP, filenameF)
            findP = 1;
        end %if strcmp
        
    end %for f
    
    %if not found, remove
    if findP == -1
        indexrem = [indexrem p];
    end %if findP
        
end %for p


%remove indexrem
problem(indexrem) = [];

%get labels
labels = [problem.class];

%get num of images
numImagesAll = numel(problem);




