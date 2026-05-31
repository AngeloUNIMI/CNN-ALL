function [deblurred_image] = deBlur1shotMaxPol(image_scan_original, params)

%M. S. Hosseini and K. N. Plataniotis, 
%"Convolutional Deblurring for Natural Imaging," 
%in IEEE Transactions on Image Processing, vol. 29, pp. 250-264, 2020.

image_scan_original = im2double(image_scan_original);
[N_1, N_2, N_3] = size(image_scan_original);

%
[h_psf, c1_estimate, c2_estimate, alpha_estimate, amplitude_estimate] = blur_kernel_estimation(image_scan_original, params.model_type, params.scale);

%
[deblurring_kernel] = deblurring_kernel_estimation(h_psf, params.model_type);

%
[deblurred_image] = OneShotMaxPol(image_scan_original, deblurring_kernel, params.model_type, alpha_estimate, c1_estimate, h_psf, params.significancy);





