function [t3] = imrotateAng(im, rotMethod, pad, padMethod, angle)

%pad
t = padarray(im, pad, padMethod);

%get center
center = round([size(t,1) / 2, size(t,2) / 2]);

%rotate around center
t2 = rotateAround(t, center(1), center(2), angle, rotMethod);

%get crop area
rect = [pad(1)+1, pad(2)+1, size(im,2)-1, size(im,1)-1];

%crop
t3 = imcrop(t2, rect);

