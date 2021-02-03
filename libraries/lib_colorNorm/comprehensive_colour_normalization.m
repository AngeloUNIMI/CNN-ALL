function normalized = comprehensive_colour_normalization(cube,varargin)
% NORMALIZED = comprehensive_colour_normalization(CUBE,THRESHOLD)
%
% This function normalizes images for both illumination intensity and
% illumination color effects. It can be applied to images with an
% 'unlimited' number of color channels. Ranging from RGB to hyperspectral 
% imagery.
%
% The input of this function is given as follows:
%
% CUBE is a MxNxL matrix in which MxN is the spatial domain and L is the
% spectral (color) domain.
%
% The function iterates until the mean square root difference between two
% consecutive steps is smaller than THRESHOLD. When threshold is not
% given it is set to 10^-12 by default.
%
% The output of this function is given by:
%
% NORMALIZED is a MxNxL matrix in which MxN is the spatial domain and L is
% the spectral (color) domain. For each pixel (i,j) mean(Mi,Nj,:) is
% approximately 1. For each wavelength/color (k) mean(:,:,Lk) is
% approximately 1 as well.
%
% This algorithm is based on the folowing paper:
% ------
% Finlayson, G., Schiele, B., & Crowley, J. (1998). 
% Comprehensive Colour Image Normalization. 
% Computer Vision—ECCV’98, 1406, 475–490. 
% https://doi.org/10.1007/BFb0055655
% ------
%
% Script written by (Niels) N.W. Schurink, 23-11-2016, master 3 student 
% Technical Medicine, University of Twente, the Netherlands, during
% master thesis at Netherlands Cancer Institute - Antoni van Leeuwenhoek

% Check the input of the function
p = inputParser;

numchk = {'numeric'};
nempty = {'nonempty'};
threeD = {'3d'};

% the input should have 3 dimensions, should be numeric and non-empty
addRequired(p,'cube',@(x)validateattributes(x,numchk,{nempty{:},threeD{:}}))

% a threshold value should be numeric and non-empty if given
addOptional(p,'threshold',1,@(x)validateattributes(x,numchk,nempty))

parse(p,cube,varargin{:})

cube = double(cube);

switch nargin
    case 1
        % set threshold if not given
        threshold = 10^-12;
    case 2
        % set threshold to user defined threshold
        threshold = varargin{1};
    otherwise
        error('Wrong number of input arguments')
end

% Initialize difference variable
difference = 1;

% Initialize 'Previous image'
Iprevious = cube;

% While the difference between itterations is larger than threshold
while difference>threshold

    % Normalize color intensities
    Inext = norm_intensity(Iprevious);
    
    % Normalize illumination
    Inext = norm_illumination(Inext);
    
    % Calculate difference between previous and next itteration
    difference = sqrt(sum(Iprevious(:).^2))-sqrt(sum(Inext(:).^2));

    % Set previous image to next image
    Iprevious = Inext;
end

% Return the normalized image
normalized = Iprevious;


%% Helper functions
%-----------------------------------------------


function output =  norm_intensity(input)
% output =  norm_intensity(input)
%
% This function normalizes the color intensities by dividing each color 
% value by the sum of all color values. In the case of an RGB image
%
% R = R/(R+G+B), G = G/(R+G+B), B = B/(R+G+B)
%
% However, this function is generalized for an 'unlimited' number of colors
% Let C_i be an arbitrary color and let the image have L colors in total
% then each color C_i is normalized by
%
% C_i = C_i/(C_1+C_2+...+C_L-1+C_L)
    output = bsxfun(@rdivide,input,sum(input,3));
    
    
function output = norm_illumination(input)
% output = norm_illumination(input)
%
% This function normalizes for the color spectrum of the illuminant of the
% scene. Suppose again that we have an RGB image. If (r1,g1,b1) and
% (r2,g2,b2) are the responses of two points in the image under one colour
% of light then (k*r1,l*g1,m*b1) and (k*r2,l*g2,m*b2) are the responses of
% these same point under a different color of light. It can be easily shown
% that we can cancel the factors (k, l and m):
%
% ( 2r1/(r1+r2), 2g1/(r1+r2), 2b1/(r1+r2) ) and
%
% ( 2r2/(r1+r2), 2g2/(r1+r2), 2b2/(r1+r2) )
%
% If we have N pixels in total and L colors in total. Let C(m,n,l) denote a
% pixel value at position (m,n) and color (c). We can then show that the
% illumination normalized image is given by
%
% ( N*C(m,n,1)/sum(C(m,n,1)), N*C(m,n,...)/sum(C(m,n,...)), N*C(m,n,L)/sum(C(m,n,L)) )

    output = size(input,1)*size(input,2)*bsxfun(@rdivide,input,sum(sum(input,1),2));