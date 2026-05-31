function [V, NumFiltersInit, retainedVar] = PCA_FilterBank(InImg, PatchSize, NumFilters, stage, param, numCoresFeatExtr) 
% =======INPUT=============
% InImg            Input images (cell structure)  
% PatchSize        the patch size, asumed to an odd number.
% NumFilters       the number of PCA filters in the bank.
% =======OUTPUT============
% V                PCA filter banks, arranged in column-by-column manner
% =========================

% addpath('./Utils')

% to efficiently cope with the large training samples, if the number of training we randomly subsample 10000 the
% training set to learn PCA filter banks
ImgZ = length(InImg);
MaxSamples = 100000;
NumRSamples = min(ImgZ, MaxSamples); 
RandIdx = randperm(ImgZ);
RandIdx = RandIdx(1:NumRSamples);

% Learning PCA filters (V)
NumChls = size(InImg{1},3);

%NORMAL
% Rx = zeros(NumChls*PatchSize^2,NumChls*PatchSize^2);
% % for i = RandIdx %1:ImgZ
% for i = 1:ImgZ
%     im = im2col_mean_removal(InImg{i},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
%     Rx = Rx + im*im'; % sum of all the input images' covariance matrix
% end
% Rx = Rx/(NumRSamples*size(im,2));



%%%
Rx_i = cell(ImgZ,1);
im = im2col_mean_removal(InImg{1},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal

% whos im

sizem = size(im,2);
% for i = RandIdx %1:ImgZ

%start_pool(numCoresFeatExtr);

parfor i = 1 : ImgZ
% for i = 1 : ImgZ
    im = im2col_mean_removal(InImg{i},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    Rx_i{i} = im*im'; % sum of all the input images' covariance matrix
end

Rx = zeros(NumChls*PatchSize^2,NumChls*PatchSize^2);
for i = 1:ImgZ
    Rx = Rx_i{i} + Rx;
end

Rx = Rx/(NumRSamples*sizem);




%original
% V = E(:,ind(1:NumFilters));  % principal eigenvectors 

%%%
% assignin('base', 'Rx', Rx);
% pause

%Eigenvalue decomposition
[E, D] = eig(Rx);
[Ds, ind] = sort(diag(D),'descend');

if param.useDynamicNumFilters

   %init
   NumFiltersInit_one = 1;
   %retained variance
   retainedVar_one = sum(Ds(1:NumFiltersInit_one)) / sum(Ds);
   %increase number of components if needed
   while retainedVar_one < param.RetainedVariance(stage)
       NumFiltersInit_one = NumFiltersInit_one + 1;
       retainedVar_one = sum(Ds(1:NumFiltersInit_one)) / sum(Ds);
   end %while  
   
   %try with one more
   NumFiltersInit_poss = NumFiltersInit_one + 1;
   retainedVar_poss = sum(Ds(1:NumFiltersInit_poss)) / sum(Ds);
   
   diff_one = abs(retainedVar_one - param.RetainedVariance(stage));
   diff_poss = abs(retainedVar_poss - param.RetainedVariance(stage));
   
   if diff_one < diff_poss
       NumFiltersInit = NumFiltersInit_one;
   else %if diff_one < diff_poss
       NumFiltersInit = NumFiltersInit_poss;
   end %if diff_one < diff_poss
   
else%if PCANet.useDynamicNumFilters
    NumFiltersInit = NumFilters;
end %if PCANet.useDynamicNumFilters


%Retained variance
retainedVar = sum(Ds(1:NumFiltersInit)) / sum(Ds);

%select principal components
V = E(:,ind(1:NumFiltersInit));  % principal eigenvectors


%PARALLEL
% im = cell(ImgZ, 1);
% InImg = parallel.pool.Constant(InImg);
% parfor j = 1 : length(RandIdx)
%     i = RandIdx(j) %1:ImgZ
%     im{j} = im2col_mean_removal(InImg.Value{i},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
%     Rx = Rx + im{j}*im{j}'; % sum of all the input images' covariance matrix
% end
% 
% Rx = Rx/(NumRSamples*size(im{1},2));
% [E, D] = eig(Rx);
% [~, ind] = sort(diag(D),'descend');
% V = E(:,ind(1:NumFilters));  % principal eigenvectors 



 



