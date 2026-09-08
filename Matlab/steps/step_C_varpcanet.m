function [] = step_C_varpcanet(dirTest, dirUtilities, ext, numCoresFeatExtr, numCoresKnn, fidLogs, logS, savefile, plotFigures)

%--------------------------------------
%General parameters
stepPrint = 100;
%PCA Params
run('./params/paramsPCATuning.m');


%--------------------------------------
%Dir DBs
dbname_All = { ...
    'ALL_IDB'
    };
dbname_part_All = { ...
    'ALL_IDB2'
    };
dbname_ROI_All = { ...
    'ROI_256'
    };

%nets
net_name{1} = 'AlexNet';
net_name{2} = 'VGG16';
net_name{3} = 'VGG19';
net_name{4} = 'ResNet18';
net_name{5} = 'ResNet50';
net_name{6} = 'ResNet101';
net_name{7} = 'DenseNet201';


%--------------------------------------
colorS_init = 1;
colorS_tune = 1;
colorS_test = 1;
%--------------------------------------

% processDummyDirs();



%--------------------------------------
%Loop on dbs
for db = 1 : numel(dbname_All)
    % for db = 2
    
    %Close
    close all
    pause(0.2);
    
    %DB selection
    dbname = dbname_All{db};
    dbnamePart = dbname_part_All{db};
    ROI = dbname_ROI_All{db};
    dirDB_wROI = [dirTest dbname '/' dbnamePart '/' ROI '/'];
    %dirDB_noROI = [dirWorkspace dbname '/' dbnamePart '/' ];
    
    
    %--------------------------------------
    %loop on nets
    for n = 1 : numel(net_name)
        
        %switch net
        switch n
            
            case 1
                net = alexnet;
                layer = 'fc6';
                conv_layer = 'conv5';
                
            case 2
                net = vgg16;
                layer = 'fc6';
                conv_layer = 'conv5_3';
                
            case 3
                net = vgg19;
                layer = 'fc6';
                conv_layer = 'conv5_4';
                
            case 4
                net = resnet18;
                layer = 'fc1000';
                conv_layer = 'res5b_relu';
                
            case 5
                net = resnet50;
                layer = 'fc1000';
                conv_layer = 'res5c_branch2c';
                
            case 6
                net = resnet101;
                layer = 'fc1000';
                conv_layer = 'res5c_branch2c';
                
            case 7
                net = densenet201;
                layer = 'fc1000';
                conv_layer = 'conv5_block32_2_conv';
                
        end %switch
        
        
        %--------------------------------------
        %Folder creation
        %RESULTS: dirs net
        dirResults = ['./Results/' dbname '/' dbnamePart '/' net_name{n} '/'];
        mkdir_pers(dirResults, savefile);
        %RESULTS: log file
        timeStampRaw = datestr(datetime);
        timeStamp = strrep(timeStampRaw, ':', '-');
        if savefile && logS
            logFile = [dirResults dbname '_log_' timeStamp '.txt'];
            fidLog = fopen(logFile, 'w');
            fidLogs{2} = fidLog;
        end %if savefile && log
        
        
        %--------------------------------------
        %Display
        fprintf_pers(fidLogs, '\n');
        fprintf_pers(fidLogs, '---------------\n');
        fprintf_pers(fidLogs, 'ALL-Unsharpen\n');
        fprintf_pers(fidLogs, [dbname '\n']);
        fprintf_pers(fidLogs, [dbnamePart '\n']);
        fprintf_pers(fidLogs, '---------------\n');
        fprintf_pers(fidLogs, '\n');
        
        
        %--------------------------------------
        %display
        fprintf_pers(fidLogs, '---------------\n');
        fprintf_pers(fidLogs, ['Net: ' net_name{n} '\n']);
        fprintf_pers(fidLogs, '---------------\n');
        fprintf_pers(fidLogs, '\n')
        
        
        %--------------------------------------
        %DB processing
        %Extract samples
        files = dir([dirDB_wROI '*.' ext]);
        
        %Compute labels
        [problem, labels, numImagesAll] = computeLabels(dirDB_wROI, files);
        
        
        %--------------------------------------
        %Display
        fprintf_pers(fidLogs, 'Extracting samples...\n');
        fprintf_pers(fidLogs, ['\t' num2str(numImagesAll) ' images in total\n']);
        fprintf_pers(fidLogs, '\n');
        
        
        %--------------------------------------
        %LOOP ON ITERATIONS
        %Init
        accuracy_knnAll = zeros(param.numIterations, 1);
        cmc = cell(param.numIterations, 1);
        
        %--------------------------------------
        %Compute random fold indexes
        %if outside iteration loop, random fold once (es. 10-fold)
        %[allIndexes, cvIndices] = computeAllIndexesFold(numImagesAll, labels, param);
        
        %Loop
        for r = 1 : param.numIterations
            
            
            %--------------------------------------
            %Display
            fprintf_pers(fidLogs, ['Iteration N. ' num2str(r) '\n']);
            
            
            %--------------------------------------
            %File save info
            fileSaveTest_iter = [dirResults '/results_iter_' num2str(r) '.mat'];
            
            
            %--------------------------------------
            %Compute random fold indexes
            %--10-fold
            %[indImagesTrain, indImagesTest, numImagesTrain, numImagesTest] = computeIndexesFold(cvIndices, r);
            %--2-fold if inside iteration loop, random fold each iteration (repeated 2-fold)
            [allIndexes, cvIndices] = computeAllIndexesFold(numImagesAll, labels, param);
            [indImagesTrain, indImagesTest, numImagesTrain, numImagesTest] = computeIndexesFold(cvIndices, randi(2, 1));
            %Corresponding labels
            TrnLabels = labels(indImagesTrain);
            TestLabels = labels(indImagesTest);
            
            
            %--------------------------------------
            %Display output number of images
            fprintf_pers(fidLogs, ['\t' num2str(numImagesTrain) ' images are chosen for training\n']);
            fprintf_pers(fidLogs, ['\t' num2str(numImagesTest) ' images are chosen for testing\n']);
            
            
            
            %%%%%%%%%%%%%%  TRAINING  %%%%%%%%%%%%%
            start_pool(numCoresFeatExtr);
            
            %--------------------------------------
            fprintf_pers(fidLogs, '\tTraining... \n')
            
            
            %--------------------------------------
            %Load images for training
            fprintf_pers(fidLogs, '\t\tLoading images for training... \n')
            [imagesCellTrain, filenameTrn, ~] = loadImages(files, dirDB_wROI, allIndexes, indImagesTrain, numImagesTrain, param, colorS_init, 100, dirUtilities, 0);
            imagesCellTrain = adjustFormat(imagesCellTrain);
            
            %norm
            %[imagesCellTrain, dd1, dd2] = computeNorm(imagesCellTrain);
            
            %find th_focus
            fprintf_pers(fidLogs, '\tComputing th_focus init... \n')
            
            %%%%%% QUI %%%%%%
            th_focus_init = find_th_focus(imagesCellTrain, TrnLabels, [128 128], dirUtilities, fidLogs);
            %th_focus_init = 7.3;
            
            %puliamo
            clear imagesCellTrain
            
            %tune
            %th_focus_start = round((th_focus_init - th_focus_init*percC/100)*10)/10;
            %th_focus_end = round((th_focus_init + th_focus_init*percC/100)*10)/10;
            th_focus_start = round((th_focus_init - 0.5)*10)/10;
            th_focus_end = round((th_focus_init + 0.5)*10)/10;
            %th_focus_start = th_focus_init;
            %th_focus_end = th_focus_init;
            
            %init
            accuracy_knnALLFOCUS = [];
            
            
            %--------------------------------------
            %tuning th_focus
            fprintf_pers(fidLogs, '\tTuning th_focus... \n')
            
            %loop on th_focus
            allfocuses = th_focus_start : 0.1 : th_focus_end;
            %%%%%% QUI %%%%%%
            for th_focus = allfocuses
            %for th_focus = th_focus_start
                
                %Display
                fprintf_pers(fidLogs, '\n')
                fprintf_pers(fidLogs, ['\t\tth_focus: ' num2str(th_focus) '\n']);
                
                
                fprintf_pers(fidLogs, '\t\tLoading images... \n')
                [imagesCellTrain, filenameTrn, ~] = loadImages(files, dirDB_wROI, allIndexes, indImagesTrain, numImagesTrain, param, colorS_tune, th_focus, dirUtilities, 0);
                imagesCellTrain = adjustFormatForPCANet(imagesCellTrain, [128, 128]);
                
                %--------------------------------------
                %PCANet Training
                %1 layer: PCA filters
                [V, PCANet] = trainPCANet(imagesCellTrain, PCANet, fidLogs, param, numCoresFeatExtr);
                %Feature extraction
                fprintf_pers(fidLogs, '\t\tFeature extraction... \n')
                [ftrain_all, numFeaturesTrain] = featExtrGaborAdapt(imagesCellTrain, V, PCANet, [], param, numImagesTrain, stepPrint);
                
                %size
                sizeTrain = size(ftrain_all, 2);
                
                %--------------------------------------
                %performance
                fprintf_pers(fidLogs, '\t\tClassification - original... \n')
                errorStruct_original_temp = computeClassificationPerformance(numFeaturesTrain, sizeTrain, ftrain_all, TrnLabels, stepPrint, numCoresKnn, fidLogs, param);
                
                %compute cmc
                [cmc_original_temp, cmc_sum_original_temp] = computeCMC(errorStruct_original_temp.distMatrixTest, TrnLabels, ['cmc original temp iteration ' num2str(r)], 0);
                
                %Error metrics
                fprintf_pers(fidLogs, ['\t\tTraining accuracy (perc. of correctly classified samples, at iteration n. ' num2str(r) '): %s%%\n'], num2str(errorStruct_original_temp.accuracy_knn*100));
                fprintf_pers(fidLogs, ['\t\tAUC of CMC (at iteration n. ' num2str(r) '): %s\n'], num2str(cmc_sum_original_temp));
                
                
                %assign
                accuracy_knnALLFOCUS = [accuracy_knnALLFOCUS errorStruct_original_temp.accuracy_knn];
                %accuracy_knnALLFOCUS = [accuracy_knnALLFOCUS cmc_sum_original_temp]; %maximize AUC of CMC
                
                
            end %th_focus
            
            %Puliamo
            clear imagesCellTrain ftrain_all
            
            
            
            
            
            %%%%%%%%%%%%%%  APPLY BEST FOCUS  %%%%%%%%%%%%%
            fprintf_pers(fidLogs, '\n')
            fprintf_pers(fidLogs, '\tApply best focus\n')
            %best focusc
            %[maxAcc, i_best_focus] = max(accuracy_knnALLFOCUS);
            %i_best_focus = i_best_focus(1);
            [sortA, isort] = sort(accuracy_knnALLFOCUS);
            i_best_focus = isort(end); %highest th_focus that gives the best result
            best_th_focus = allfocuses(i_best_focus);
            fprintf_pers(fidLogs, ['\t\tBest focus: ' num2str(best_th_focus) ' \n']);
            
            %training data
            fprintf_pers(fidLogs, '\t\tLoading training images - original... \n')
            [imagesCellTrain_original, ~, ~] = loadImages(files, dirDB_wROI, allIndexes, indImagesTrain, numImagesTrain, param, colorS_test, 100, dirUtilities, 0);
            imagesCellTrain_original = adjustFormat(imagesCellTrain_original);
            %norm
            %[imagesCellTrain_original, meanA_original, stdA_original] = computeNorm(imagesCellTrain_original);
            fprintf_pers(fidLogs, '\t\tLoading training images - unsharpened... \n')
            [imagesCellTrain_unsharp, ~, ~] = loadImages(files, dirDB_wROI, allIndexes, indImagesTrain, numImagesTrain, param, colorS_test, best_th_focus, dirUtilities, 0);
            imagesCellTrain_unsharp = adjustFormat(imagesCellTrain_unsharp);
            %norm
            %[imagesCellTrain_unsharp, meanA_unsharp, stdA_unsharp] = computeNorm(imagesCellTrain_unsharp);
            
            %testing data
            fprintf_pers(fidLogs, '\t\tLoading testing images - original... \n')
            [imagesCellTest_original, ~, ~] = loadImages(files, dirDB_wROI, allIndexes, indImagesTest, numImagesTest, param, colorS_test, 100, dirUtilities, 0);
            [imagesCellTest_original, meanAll_test_original] = adjustFormat(imagesCellTest_original);
            %imagesCellTest_original = applyNorm(imagesCellTest_original, meanA_original, stdA_original);
            fprintf_pers(fidLogs, '\t\tLoading testing images - unsharpened... \n')
            [imagesCellTest_unsharp, filenameTest, indexes_test_imUnsharpened] = loadImages(files, dirDB_wROI, allIndexes, indImagesTest, numImagesTest, param, colorS_test, best_th_focus, dirUtilities, 0);
            [imagesCellTest_unsharp, meanAll_test_unsharp] = adjustFormat(imagesCellTest_unsharp);
            %imagesCellTest_unsharpened = applyNorm(imagesCellTest_unsharpened, meanA_unsharp, stdA_unsharp);
            
            
            
            %%%%%%%%%%%%%%  TESTING - PRE-TRAINED CNNs %%%%%%%%%%%%%
            %pre-trained
            fprintf_pers(fidLogs, '\t\tPretrained CNNs... \n')
            
            
            %Feature extraction - ORIGINAL
            fprintf_pers(fidLogs, '\t\t\tFeature extraction - original... \n')
            ftrain_all_original = feature_extraction_cnn(imagesCellTrain_original, net, layer, colorS_test);
            ftest_all_original = feature_extraction_cnn(imagesCellTest_original, net, layer, colorS_test);
            
            %Feature extraction - UNSHARP
            fprintf_pers(fidLogs, '\t\t\tFeature extraction - unsharp... \n')
            ftrain_all_unsharp = feature_extraction_cnn(imagesCellTrain_unsharp, net, layer, colorS_test);
            ftest_all_unsharp = feature_extraction_cnn(imagesCellTest_unsharp, net, layer, colorS_test);
            
            %size
            numFeatures = size(ftest_all_unsharp, 1);
            sizeTest = size(ftest_all_unsharp, 2);
            
            
            %--------------------------------------
            %Classification performance
            %Original
            %fprintf_pers(fidLogs, '\t\t\tClassification - original... \n')
            %errorStruct_pretrained_original(r) = computeClassificationPerformance(numFeatures, sizeTest, ftest_all_original, TestLabels, stepPrint, numCoresKnn, fidLogs, param);
            errorStruct_pretrained_original(r) = computeClassificationPerformanceTrainTest(numFeatures, sizeTest, ftrain_all_original, ftest_all_original, TrnLabels, TestLabels, stepPrint, numCoresKnn, fidLogs, param);
            %Unsharp
            %fprintf_pers(fidLogs, '\t\t\tClassification - unsharp... \n')
            %errorStruct_pretrained_unsharp(r) = computeClassificationPerformance(numFeatures, sizeTest, ftest_all_unsharp, TestLabels, stepPrint, numCoresKnn, fidLogs, param);
            errorStruct_pretrained_unsharp(r) = computeClassificationPerformanceTrainTest(numFeatures, sizeTest, ftrain_all_unsharp, ftest_all_unsharp, TrnLabels, TestLabels, stepPrint, numCoresKnn, fidLogs, param);
            
            %puliamo
            clear ftest_all_original ftest_all_unsharp
            
            %compute cmc
            [cmc1, cmc_sum1] = computeCMC_trainTest(errorStruct_pretrained_original(r).distMatrixTest, TestLabels, ['cmc original iteration ' num2str(r)], plotFigures);
            [cmc2, cmc_sum2] = computeCMC_trainTest(errorStruct_pretrained_unsharp(r).distMatrixTest, TestLabels, ['cmc unsharp iteration ' num2str(r)], plotFigures);
            cmc_original{r} = cmc1;
            cmc_sum_original{r} = cmc_sum1;
            cmc_unsharp{r} = cmc2;
            cmc_sum_unsharp{r} = cmc_sum2;
            errorStruct_pretrained_original(r).rank5 = cmc1(5);
            errorStruct_pretrained_unsharp(r).rank5 = cmc2(5);
            
            %Display
            fprintf_pers(fidLogs, '\n')
            fprintf_pers(fidLogs, ['\tPretrained - Accuracy original (at iteration n. ' num2str(r) '): %s%%\n'], num2str(errorStruct_pretrained_original(r).accuracy_knn*100));
            fprintf_pers(fidLogs, ['\tPretrained - Accuracy unsharp (at iteration n. ' num2str(r) '): %s%%\n'], num2str(errorStruct_pretrained_unsharp(r).accuracy_knn*100));
            fprintf_pers(fidLogs, ['\tPretrained - Rank 5 accuracy original (at iteration n. ' num2str(r) '): %s%%\n'], num2str(cmc1(5)*100));
            fprintf_pers(fidLogs, ['\tPretrained - Rank 5 accuracy unsharp (at iteration n. ' num2str(r) '): %s%%\n'], num2str(cmc2(5)*100));
            
            pause(0.1)
            
            
            %%%%%%%%%%%%%%  TESTING - FINE TUNING CNNs %%%%%%%%%%%%%
            %pre-trained
            fprintf_pers(fidLogs, '\n')
            fprintf_pers(fidLogs, '\tFine tuning CNNs... \n')
            %fprintf_pers(fidLogs, '\t\t\tTraining... \n')
            
            pixelRange = [-30 30];
            rotRange = [-180 180];
            
            imageAugmenter = imageDataAugmenter( ...
                'RandXReflection', true, ...
                'RandYReflection', true, ...
                'RandRotation', rotRange);
            %'RandXTranslation', pixelRange, ...
            %'RandYTranslation', pixelRange ...
            
            
            numClasses = numel(unique(labels));
            inputSize = net.Layers(1).InputSize;
            %             layersTransfer = net.Layers(1:end-3);
            %             layers = [
            %                 layersTransfer
            %                 fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
            %                 softmaxLayer
            %                 classificationLayer];
            
            %change last layers
            lgraph = replaceLayers(net, numClasses);
            
            %options
            options = trainingOptions('sgdm', ...
                'MiniBatchSize', 20, ... %128 20
                'MaxEpochs', 100, ...
                'InitialLearnRate', 1e-4, ...
                'Shuffle', 'every-epoch', ...   'never'
                'ValidationFrequency', 3, ...
                'Verbose', false, ...
                'Plots', 'none'); %   'training-progress'
            
            fprintf_pers(fidLogs, '\t\tTraining original... \n')
            %netTransfer_original = fineTuneCNN(imagesCellTrain_original, TrnLabels, './dummy_train_original/', inputSize, imageAugmenter, layers, options);
            netTransfer_original = fineTuneCNN(imagesCellTrain_original, TrnLabels, './dummy_train_original/', inputSize, imageAugmenter, lgraph, options);
            
            fprintf_pers(fidLogs, '\t\tTraining unsharp... \n')
            %netTransfer_unsharp = fineTuneCNN(imagesCellTrain_unsharp, TrnLabels, './dummy_train_unsharp/', inputSize, imageAugmenter, layers, options);
            netTransfer_unsharp = fineTuneCNN(imagesCellTrain_unsharp, TrnLabels, './dummy_train_unsharp/', inputSize, imageAugmenter, lgraph, options);
            
            %
            %fprintf_pers(fidLogs, '\t\t\tTesting... \n');
            
            %cm
            fprintf_pers(fidLogs, '\t\tTesting original... \n')
            errorStruct_finetune_original(r) = computeClassPerformanceFineTuneCNN(imagesCellTest_original, TestLabels, './dummy_test_original/', inputSize, netTransfer_original, fidLogs);
            
            fprintf_pers(fidLogs, '\t\tTesting unsharp... \n')
            errorStruct_finetune_unsharp(r) = computeClassPerformanceFineTuneCNN(imagesCellTest_unsharp, TestLabels, './dummy_test_unsharp/', inputSize, netTransfer_unsharp, fidLogs);
            
            %Display
            fprintf_pers(fidLogs, ['\tFine tuning - Accuracy original (at iteration n. ' num2str(r) '): %s%%\n'], num2str(errorStruct_finetune_original(r).accuracy_knn*100));
            fprintf_pers(fidLogs, ['\tFine tuning - Accuracy unsharp (at iteration n. ' num2str(r) '): %s%%\n'], num2str(errorStruct_finetune_unsharp(r).accuracy_knn*100));
            
            
            %--------------------------------------
            %Save
            if savefile
                save(fileSaveTest_iter, 'errorStruct_pretrained_original', 'errorStruct_pretrained_unsharp', 'errorStruct_finetune_original', 'errorStruct_finetune_unsharp', 'cmc_original', 'cmc_unsharp');
            end %if savefile
            
            
            %--------------------------------------
            %Display progress
            fprintf_pers(fidLogs, '\n');
            
            
            %GRAD-CAM
            fprintf_pers(fidLogs, 'Grad-CAM\n');
            fprintf_pers(fidLogs, '\n');
            dirGcam = [dirResults 'gcam_iter_' num2str(r) '/'];
            mkdir_pers(dirGcam, savefile);
            [imagesCellTest_original, ~, ~] = loadImages(files, dirDB_wROI, allIndexes, indImagesTest, numImagesTest, param, colorS_test, 100, dirUtilities, 0);
            [imagesCellTest_unsharp, ~, ~] = loadImages(files, dirDB_wROI, allIndexes, indImagesTest, numImagesTest, param, colorS_test, best_th_focus, dirUtilities, 0);
            computeGradCam(imagesCellTest_original, imagesCellTest_unsharp, meanAll_test_original, meanAll_test_unsharp, indexes_test_imUnsharpened, ...
               netTransfer_original, netTransfer_unsharp, conv_layer, inputSize, filenameTest, TestLabels, dirGcam);
            %computeGradCam2(imagesCellTest_original, imagesCellTest_unsharp, meanAll_test_original, meanAll_test_unsharp, indexes_test_imUnsharpened, ...
               %netTransfer_original, netTransfer_unsharp, conv_layer, inputSize, filenameTest, TestLabels, dirGcam);
            
            
            
            
            %Puliamo
            clear imagesCellTrain_original imagesCellTrain_unsharpenend imagesCellTest_original imagesCellTest_unsharpened netTransfer_original netTransfer_unsharp layers
            
            
        end %for r = 1 : param.numIterations
        
        
        close all
        pause(0.1)
        
        %display
        fprintf_pers(fidLogs, '\n');
        
        
        
        %--------------------------------------
        %Average classification performance
        %PRETRAINED
        %Error metrics
        fprintf_pers(fidLogs, '\n');
        %original
        fprintf_pers(fidLogs, 'Pretrained - Original\n')
        stampaErrors(errorStruct_pretrained_original, fidLogs);
        fprintf_pers(fidLogs, '\tRank 5 accuracy (mean; std): %s%%; %s%% \n', num2str(mean([errorStruct_pretrained_original.rank5])*100), num2str(std([errorStruct_pretrained_original.rank5])*100));
        %unsharp
        fprintf_pers(fidLogs, 'Pretrained - Unsharp\n')
        stampaErrors(errorStruct_pretrained_unsharp, fidLogs);
        fprintf_pers(fidLogs, '\tRank 5 accuracy (mean; std): %s%%; %s%% \n', num2str(mean([errorStruct_pretrained_unsharp.rank5])*100), num2str(std([errorStruct_pretrained_unsharp.rank5])*100));
        
        %FINE TUNING
        %Error metrics
        fprintf_pers(fidLogs, '\n');
        %original
        fprintf_pers(fidLogs, 'Fine tuning - Original\n')
        stampaErrors(errorStruct_finetune_original, fidLogs);
        %unsharp
        fprintf_pers(fidLogs, 'Fine tuning - Unsharp\n')
        stampaErrors(errorStruct_finetune_unsharp, fidLogs);
        
        
        
        
        %--------------------------------------
        %Average CMC
        %original
        stampaAvgCMC(cmc_original, 'original', dirResults, savefile, plotFigures);
        %unsharp
        stampaAvgCMC(cmc_unsharp, 'unsharp', dirResults, savefile, plotFigures);
        
        
        
        
        %--------------------------------------
        %Display progress
        fprintf_pers(fidLogs, '\n');
        
        
        %--------------------------------------
        %Close file log
        if savefile && logS
            fclose(fidLog);
        end %if savefile && log
        %         delete(gcp('nocreate'));
        fclose('all');
        
        
        %close
        close all
        pause(0.1)
        
        
    end %for n
    
    
end %for db



