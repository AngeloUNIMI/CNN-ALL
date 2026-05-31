function [imColorNorm] = imNormalization(im, filename, plotta)

%color normalization
imColorNorm = comprehensive_colour_normalization(im);

%display - focus
if plotta
    fh = fsfigure;
    imshow(imresize(imColorNorm, 4))
    title(['Im: ' filename ' ; color normalization'], 'Interpreter', 'none')
end %if plotta