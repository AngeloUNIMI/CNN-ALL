function ftest_all = feature_extraction_cnn(imagesCellTest_original, net, layer, colorS)

im_ = imresize(imagesCellTest_original{1}, net.Layers(1).InputSize(1:2));
if colorS == 0
    im_ = repmat(im_, [1 1 3]); %if gray, need to make it 3 channels
end %if colorS
temp = activations(net, im_, layer, 'OutputAs', 'rows');
ftest_all = zeros(numel(temp), numel(imagesCellTest_original));
ftest_all(:, 1) = temp;

for ind_im = 2 : numel(imagesCellTest_original)
    im_ = imresize(imagesCellTest_original{ind_im}, net.Layers(1).InputSize(1:2));
    if colorS == 0
        im_ = repmat(im_, [1 1 3]); %if gray, need to make it 3 channels
    end %if colorS
    ftest_all(:, ind_im) = activations(net, im_, layer, 'OutputAs', 'rows');
end %for ind_im
