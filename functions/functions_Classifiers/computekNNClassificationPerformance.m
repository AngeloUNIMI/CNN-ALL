function [TestPCAOutput, distMatrix] = computekNNClassificationPerformance(ftestPCA_all, TestPCALabels, sizeTest, stepPrint, numCoresKnn, param)


%Distance matrix
if strcmp(param.matchDistance, 'chisq')
    distMatrix = sc_pdist2(ftestPCA_all, ftestPCA_all, 'chisq');
elseif strcmp(param.matchDistance, 'euclidean')
    %fast Euclidean distance
    distMatrix = full(fastEuclideanDistance(ftestPCA_all, ftestPCA_all));
else %if strcmp
    distMatrix = pdist2(ftestPCA_all', ftestPCA_all', param.matchDistance);
end %if strcmp

%modo parallelo per knn (k = 1) classification
%leave-one-out
%we use knn_neighbors + 1 because otherwise it would find the same vector
%in this way we choose the second neighbor (which is the actual first neighbor)
%(we use the entire feature vector for all samples)
%loop on test samples
%init
TestPCAOutput = zeros(sizeTest, 1);

%parallel vars init
% ftestPCA_all = parallel.pool.Constant(ftestPCA_all);
knnDistancePar = param.knnDistance;
matchDistance = param.matchDistance;
numkPar = param.knn_neighbors;

% start_pool(numCoresKnn);

% parfor g = 1 : sizeTest
for g = 1 : sizeTest
    
    %get id of current worker
    t = getCurrentTask();
    
    %display progress
    if mod(g, stepPrint) == 0
        %fprintf(1, ['\t\tCore ' num2str(t.ID) ': ' num2str(g) ' / ' num2str(sizeTest) '\n'])
        fprintf(1, ['\t\t' num2str(g) ' / ' num2str(sizeTest) '\n'])
    end %if mod(i, 100) == 0
    
    if strcmp(knnDistancePar, matchDistance)
        %we can re-use the distance matrix
        distV = distMatrix(g, :);
        sortV = sort(distV, 'ascend');
        minD = sortV(2); %the first will be 0
        idx = find(distV == minD);
        idx = idx(1); %se dovessero essercene altri a pari merito  
        
    else %if strcmp(param.knnDistance, param.matchDistance)
        %we use + 1
        %idx = knnsearch(ftestPCA_all.Value', ftestPCA_all.Value(:, g)', 'K', numkPar + 1, 'Distance', knnDistancePar);
        idx = knnsearch(ftestPCA_all', ftestPCA_all(:, g)', 'K', numkPar + 1, 'Distance', knnDistancePar);
        idx(idx == g)  = []; %il più vicino è il vettore stesso, lo togliamo
        idx = idx(1); %se dovessero essercene altri a pari merito
    end %if strcmp(param.knnDistance, param.matchDistance)
    
    TestPCAOutput(g) = idx;
    
end %for g

%mettiamo le labels al posto degli indici trovati
for g = 1 : sizeTest
    TestPCAOutput(g) = TestPCALabels(TestPCAOutput(g));
end %for g


