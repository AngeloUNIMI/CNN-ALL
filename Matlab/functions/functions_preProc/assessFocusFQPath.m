function [scoreFocus] = assessFocusFQPath(im, filename, dirUtilities, plotta)

%M. S. Hosseini, J. A. Z. Brawley-Hayes, Y. Zhang, L. Chan, K. N. Plataniotis and S. Damaskinos,
%"Focus Quality Assessment of High-Throughput Whole Slide Imaging in Digital Pathology,"
%in IEEE Transactions on Medical Imaging, 2019

if size(im, 3) == 3
    imGray = rgb2gray(im);
end %if size
image = im2double(imGray);

%Load kernel and identify image blur type
fileLoad = load([dirUtilities 'FQPath_kernel.mat']);
kernel_sheet = fileLoad.FQPath_kernel{:};

%get score
scoreFocus = FQPath(image, kernel_sheet);

%display - focus
if plotta
    fh = fsfigure;
    imshow(imresize(im, 4))
    title(['Im: ' filename ' ; focus score: ' num2str(scoreFocus)], 'Interpreter', 'none')
end %if plotta