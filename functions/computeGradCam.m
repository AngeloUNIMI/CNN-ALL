function computeGradCam(imagesCellTest_original, imagesCellTest_unsharpened, meanAll_test_original, meanAll_test_unsharp, ...
    indexes_test_imUnsharpened, netTransfer_original, netTransfer_unsharp, convLayerName, inputSize, filenameTest, TestLabels, dirResults)

lgraph = layerGraph(netTransfer_unsharp.Layers);
lgraph = removeLayers(lgraph, lgraph.Layers(end).Name);
dlnet = dlnetwork(lgraph);
% softmaxName = 'softmax';
softmaxName = 'prob';
% convLayerName = 'conv5_3';

%loop on imgs
%for ind_im = 1 : numel(imagesCellTest_unsharpened)
for ind_im = 1 : 10 %only some images
    
    %only images unsharpened
    if ~indexes_test_imUnsharpened(ind_im)
        continue
    end %if indexes
    
    img = imagesCellTest_unsharpened{ind_im};
    
    %img = img + meanAll_test_unsharp(ind_im);
    
    img = imresize(img, inputSize(1:2));
    %img = im2uint8(img);
    
    [classfn, score] = classify(netTransfer_unsharp, img);
    dlImg = dlarray(single(img), 'SSC');
    [convMap, dScoresdMap] = dlfeval(@gradcam, dlnet, dlImg, softmaxName, convLayerName, classfn);
    gradcamMap = sum(convMap .* sum(dScoresdMap, [1 2]), 3);
    gradcamMap = extractdata(gradcamMap);
    gradcamMap = rescale(gradcamMap);
    gradcamMap = imresize(gradcamMap, inputSize(1:2), 'Method', 'bicubic');
    
    imOriginal = imagesCellTest_original{ind_im};
    
    %imOriginal = imOriginal + meanAll_test_original(ind_im);    

    imOriginal = imresize(imOriginal, inputSize(1:2));
    %imOriginal = im2uint8(imOriginal);
    
    [classfn_original, score_original] = classify(netTransfer_unsharp, imOriginal);
    dlImg = dlarray(single(imOriginal), 'SSC');
    [convMap, dScoresdMap] = dlfeval(@gradcam, dlnet, dlImg, softmaxName, convLayerName, classfn_original);
    gradcamMap_original = sum(convMap .* sum(dScoresdMap, [1 2]), 3);
    gradcamMap_original = extractdata(gradcamMap_original);
    gradcamMap_original = rescale(gradcamMap_original);
    gradcamMap_original = imresize(gradcamMap_original, inputSize(1:2), 'Method', 'bicubic');
    
    
    resVisFact = 1;
    
    fsfigure(11);
    
    subplot(2,2,1)
    imshow(imresize(imOriginal, resVisFact));
    title(sprintf('%s; Real: %s; Class.: %s (score: %.2f)', filenameTest{ind_im}, num2str(TestLabels(ind_im)), classfn_original, score_original(classfn_original)), 'Interpreter', 'none');
    subplot(2,2,2)
    imshow(imresize(imOriginal, resVisFact));
    hold on;
    imagesc(imresize(gradcamMap_original, resVisFact),'AlphaData',0.5);
    colormap jet
    hold off;
    title([filenameTest{ind_im} '; Grad-CAM'], 'Interpreter', 'none');
    
    subplot(2,2,3)
    imshow(imresize(img, resVisFact));
    title(sprintf('%s; Real: %s; Class.: %s (score: %.2f)', filenameTest{ind_im}, num2str(TestLabels(ind_im)), classfn, score(classfn)), 'Interpreter', 'none');
    subplot(2,2,4)
    imshow(imresize(img, resVisFact));
    hold on;
    imagesc(imresize(gradcamMap, resVisFact),'AlphaData',0.5);
    colormap jet
    hold off;
    title([filenameTest{ind_im} '; Grad-CAM'], 'Interpreter', 'none');
    
    set(gcf, 'color', 'w');
    
    export_fig(gcf, [dirResults, filenameTest{ind_im}]);
    
    if 0
        %write separate
        resVisFact = 5;
        
        [C, ind] = strsplit(filenameTest{ind_im}, '.');
        fileNameGradCAMOriginal = [C{1} '_gradcamOriginal.png'];
        fileNameGradCAMsharp = [C{1} '_gradcamUnsharp.png'];
        
        fileNameGradimOriginal = [C{1} '_Original.png'];
        fileNameGradimsharp = [C{1} '_Unsharp.png'];
        
        imwrite(imOriginal, [dirResults, fileNameGradimOriginal]);
        imwrite(img, [dirResults, fileNameGradimsharp]);
        
        fsfigure(12);
        imshow(imresize(imOriginal, resVisFact));
        hold on;
        imagesc(imresize(gradcamMap_original, resVisFact),'AlphaData',0.5);
        colormap jet
        hold off;
        %title([filenameTest{ind_im} '; Grad-CAM'], 'Interpreter', 'none');
        export_fig(gcf, [dirResults, fileNameGradCAMOriginal]);
        
        fsfigure(13);
        imshow(imresize(img, resVisFact));
        hold on;
        imagesc(imresize(gradcamMap, resVisFact),'AlphaData',0.5);
        colormap jet
        hold off;
        %title([filenameTest{ind_im} '; Grad-CAM'], 'Interpreter', 'none');
        export_fig(gcf, [dirResults, fileNameGradCAMsharp]);
    end
    
    pause(0.1)
    %pause
    
end %for ind_im


