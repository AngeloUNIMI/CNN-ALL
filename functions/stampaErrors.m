function stampaErrors(errorStruct, fidLogs)

err_knn_percent_mean = mean([errorStruct.err_knn_percent]);
err_knn_percent_std = std([errorStruct.err_knn_percent]);

TPR_mean = mean([errorStruct.TPR]);
TPR_std = std([errorStruct.TPR]);

TNR_mean = mean([errorStruct.TNR]);
TNR_std = std([errorStruct.TNR]);

FPR_mean = mean([errorStruct.FPR]);
FPR_std = std([errorStruct.FPR]);

FNR_mean = mean([errorStruct.FNR]);
FNR_std = std([errorStruct.FNR]);

sens_mean = mean([errorStruct.sens]);
sens_std = std([errorStruct.sens]);

spec_mean = mean([errorStruct.spec]);
spec_std = std([errorStruct.spec]);

accuracy_knnMean = mean([errorStruct.accuracy_knn]);
accuracy_knnStd = std([errorStruct.accuracy_knn]);


%Display
fprintf_pers(fidLogs, '\tErr (mean; std): %s%%; %s%% \n', num2str(err_knn_percent_mean*100), num2str(err_knn_percent_std*100));
fprintf_pers(fidLogs, '\tTP (mean; std): %s%%; %s%% \n', num2str(TPR_mean*100), num2str(TPR_std*100));
fprintf_pers(fidLogs, '\tTN (mean; std): %s%%; %s%% \n', num2str(TNR_mean*100), num2str(TNR_std*100));
fprintf_pers(fidLogs, '\tFP (mean; std): %s%%; %s%% \n', num2str(FPR_mean*100), num2str(FPR_std*100));
fprintf_pers(fidLogs, '\tFN (mean; std): %s%%; %s%% \n', num2str(FNR_mean*100), num2str(FNR_std*100));
fprintf_pers(fidLogs, '\tSensitivity (mean; std): %s%%; %s%% \n', num2str(sens_mean*100), num2str(sens_std*100));
fprintf_pers(fidLogs, '\tSpecificity (mean; std): %s%%; %s%% \n', num2str(spec_mean*100), num2str(spec_std*100));
fprintf_pers(fidLogs, '\tAccuracy (mean; std): %s%%; %s%% \n', num2str(accuracy_knnMean*100), num2str(accuracy_knnStd*100));
%fprintf_pers(fidLogs, '\n');

