function stampaAvgCMC(cmc, titleS, dirResults, savefile, plotta)

padExtra = 30;
maxSize = -1;
for c = 1 : numel(cmc)
    sizeCmc = numel(cmc{c});
    if sizeCmc > maxSize
        maxSize = sizeCmc;
    end %if sizeCmc
end %for c
%init
cmcMean = zeros(padExtra, 1);
%-pad to max size
for c = 1 : numel(cmc)
    cmcMean = cmcMean + padarray(cmc{c}, padExtra-numel(cmc{c}), 1, 'post');
end %for c
%-average
cmcMean = cmcMean ./ numel(cmc);
%-plot
if plotta
    fsfigure;
    fs = 24;
    plot(cmcMean, 'r-h', 'linewidth', 2, 'Markersize', 15);
    xlabel('Rank', 'fontsize', fs)
    ylabel('Identification Rate (%)', 'fontsize',fs)
    grid on
    title(titleS, 'fontsize', fs)
    hold on
    set(gca, 'fontsize', fs)
    set(gcf, 'color', 'w');
    axis([0 padExtra 0.80 1]);
    %save
    if savefile
        export_fig(gcf, [dirResults 'cmc_' titleS '.jpg']);
    end %if save
end %if plot
