function [TestPCAOutput, distMatrix] = computekNNClassificationPerformanceTrainTest(ftrain_all, ftestPCA_all, TrnPCALabels, TestPCALabels, sizeTest, stepPrint, numCoresKnn, param)


%Distance matrix
distMatrix = full(fastEuclideanDistance(ftestPCA_all, ftrain_all));

% whos ftrain_all ftestPCA_all distMatrix

%modo parallelo per knn (k = 1) classification
%leave-one-out
%we use knn_neighbors + 1 because otherwise it would find the same vector
%in this way we choose the second neighbor (which is the actual first neighbor)
%(we use the entire feature vector for all samples)
%loop on test samples
%init
TestPCAOutput = zeros(sizeTest, 1);

% start_pool(numCoresKnn);

% parfor g = 1 : sizeTest
for g = 1 : sizeTest
    
    %get id of current worker
    %t = getCurrentTask();
    
    %display progress
    if mod(g, stepPrint) == 0
        %fprintf(1, ['\t\tCore ' num2str(t.ID) ': ' num2str(g) ' / ' num2str(sizeTest) '\n'])
        fprintf(1, ['\t\t' num2str(g) ' / ' num2str(sizeTest) '\n'])
    end %if mod(i, 100) == 0
    
    %we can re-use the distance matrix
    distV = distMatrix(g, :);
    sortV = sort(distV, 'ascend');
    %minD = sortV(2); %the first will be 0
    minD = sortV(1); %the first will be 0
    idx = find(distV == minD);
    idx = idx(1); %se dovessero essercene altri a pari merito
    
    TestPCAOutput(g) = idx;
    
end %for g

%mettiamo le labels al posto degli indici trovati
for g = 1 : sizeTest
    %TestPCAOutput(g) = TestPCALabels(TestPCAOutput(g));
    TestPCAOutput(g) = TrnPCALabels(TestPCAOutput(g));
end %for g


