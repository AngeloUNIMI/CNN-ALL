function [indImagesTrain, indImagesTest, numImagesTrain, numImagesTest] = computeIndexesFold(cvIndices, r)

% %9/10 for training
% indImagesTest = (cvIndices == r);
% indImagesTrain = ~indImagesTest;

%1/10 for training
indImagesTrain = (cvIndices == r);
indImagesTest = ~indImagesTrain;

numImagesTrain = numel(find(indImagesTrain));
numImagesTest = numel(find(indImagesTest));