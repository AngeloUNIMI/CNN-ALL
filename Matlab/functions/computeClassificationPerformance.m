function errorStruct = computeClassificationPerformance(numFeatures, sizeTest, ftest_all, TestLabels, stepPrint, numCoresKnn, fidLogs, param)

%1-NN classifier (Nearest Neighbor)
fprintf_pers(fidLogs, '\t\tClassification... \n');
%display
fprintf_pers(fidLogs, ['\t\tNumber of features: ' num2str(numFeatures) '\n']);
fprintf_pers(fidLogs, ['\t\tNumber of samples: ' num2str(sizeTest) '\n']);
%time
tic
[TestOutput, distMatrixTest] = computekNNClassificationPerformance(ftest_all, TestLabels, sizeTest, stepPrint, numCoresKnn, param);
%Time for feature extraction
timeClass = toc;
fprintf_pers(fidLogs, ['\t\tTime for classification: ' num2str(timeClass) ' s\n']);
%Confusion matrix
C_knn = confusionmat(TestLabels, TestOutput);

%pause

%Error metrics
errorStruct = computeErrorsFromCM(C_knn);
errorStruct.distMatrixTest = distMatrixTest;
errorStruct.rank5 = [];



