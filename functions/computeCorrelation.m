function [corrFocusLabels] = computeCorrelation(focusAll, labels)

corrFocusLabels = corr2(focusAll, labels);
