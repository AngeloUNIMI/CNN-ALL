function [convMap,dScoresdMap] = gradcam(dlnet, dlImg, softmaxName, convLayerName, classfn)

[scores,convMap] = predict(dlnet, dlImg, 'Outputs', {softmaxName, convLayerName});
classScore = scores(classfn);
dScoresdMap = dlgradient(classScore,convMap);

end
