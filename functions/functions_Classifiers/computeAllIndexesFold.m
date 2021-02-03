function [allIndexes, cvIndices] = computeAllIndexesFold(numImagesAll, labels, param)

allIndexes = 1 : numImagesAll;
cvIndices = crossvalind('Kfold', labels, param.kfold);
