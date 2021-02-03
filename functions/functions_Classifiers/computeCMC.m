function [cmcPad, cmc_sum] = computeCMC(distMatrix, TestLabels, titleS, plotta)

truncCMC = 100;

clear rankV
rankV = zeros(size(distMatrix, 1), 1);
for r = 1 : size(distMatrix, 1)
    
    distV = distMatrix(r, :);
    sortV = sort(distV, 'ascend');
    sortV(1) = [];
    
    minD = sortV(1); %the first will be 0
    idx = find(distV == minD);
    idx = idx(1); %se dovessero essercene altri a pari merito
    
    rankV(r) = 1;
    
    while (TestLabels(idx) ~= TestLabels(r)) && (length(sortV) >= 2)
        
        sortV(1) = [];
        minD = sortV(1);
        idx = find(distV == minD);
        idx = idx(1); %se dovessero essercene altri a pari merito
        
        rankV(r) = rankV(r) + 1;
        
    end %while strcmp
    
    
end %for r

%
probRanks = countmember(1:max(rankV), rankV) ./ size(distMatrix, 1);
clear cmc
cmc = zeros(length(probRanks), 1);
for i = 1 : length(probRanks)
    if i == 1
        cmc(i) = probRanks(i);
    else
        cmc(i) = cmc(i-1) + probRanks(i);
    end %if i
end %for i

%truncate cmc
%cmc = cmc(1 : truncCMC);

padExtra = 30;
if numel(cmc) < padExtra
cmcPad = padarray(cmc, padExtra-numel(cmc), 1, 'post');
else
   cmcPad = cmc;
end

%cmc_sum
cmc_sum = sum(cmcPad(:));



%plot
if plotta
    fsfigure;
    fs = 24;
    plot(cmcPad, 'r-h', 'linewidth', 2, 'Markersize', 15);
    xlabel('Rank', 'fontsize', fs)
    ylabel('Identification Rate (%)', 'fontsize',fs)
    grid on
    title(titleS, 'fontsize', fs)
    hold on
    set(gca, 'fontsize', fs)
    set(gcf, 'color', 'w');
    axis([0 padExtra 0.80 1]);
end %if plot


