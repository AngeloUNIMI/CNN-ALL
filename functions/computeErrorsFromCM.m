function [errorStruct] = computeErrorsFromCM(C_knn)

%classification error
numberMisClassified = getNumberMisclassifiedSamples(C_knn);
errorStruct.err_knn_percent = numberMisClassified / sum(C_knn(:));
%TP
errorStruct.TP = C_knn(2,2);
%TN
errorStruct.TN = C_knn(1,1);
%FP
errorStruct.FP = C_knn(1,2);
%FN
errorStruct.FN = C_knn(2,1);
%sensitivity
errorStruct.sens = errorStruct.TP / (errorStruct.TP + errorStruct.FN);
%specificity
errorStruct.spec = errorStruct.TN / (errorStruct.TN + errorStruct.FP);
%TPR
errorStruct.TPR = errorStruct.TP / sum(C_knn(:));
%TNR
errorStruct.TNR = errorStruct.TN / sum(C_knn(:));
%FPR
errorStruct.FPR = errorStruct.FP / sum(C_knn(:));
%FNR
errorStruct.FNR = errorStruct.FN / sum(C_knn(:));
%accuracy
errorStruct.accuracy_knn = (sum(C_knn(:)) - numberMisClassified) / sum(C_knn(:));