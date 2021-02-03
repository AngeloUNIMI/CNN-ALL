function lgraph = replaceLayers(net, numClasses)

%https://it.mathworks.com/help/deeplearning/ug/train-deep-learning-network-to-classify-new-images.html

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer, classLayer] = findLayersToReplace(lgraph);

if isa(learnableLayer, 'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name', 'new_fc', ...
        'WeightLearnRateFactor', 20, ...
        'BiasLearnRateFactor', 20);
    
elseif isa(learnableLayer, 'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1, numClasses, ...
        'Name', 'new_conv', ...
        'WeightLearnRateFactor', 20, ...
        'BiasLearnRateFactor', 20);
end

lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnableLayer);

newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph, classLayer.Name, newClassLayer);

