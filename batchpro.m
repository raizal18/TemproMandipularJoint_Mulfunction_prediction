function results = batchpro(im)
%Image Processing Function
%
% IM      - Input image.
% RESULTS - A scalar structure with the processing results.
%

%--------------------------------------------------------------------------
% Auto-generated by imageBatchProcessor App. 
%
% When used by the App, this function will be called for every input image
% file automatically. IM contains the input image as a matrix. RESULTS is a
% scalar structure containing the results of this processing function.
%
%--------------------------------------------------------------------------



% Replace the sample below with your code----------------------------------
im=imcrop(im,[310.510000000000,1.51000000000000,479.980000000000,334.980000000000]);
if(size(im,3)==3)
    % Convert RGB to grayscale
    imgray = rgb2gray(im);
else
    imgray = im;
end

% bw = imbinarize(imgray);

results.imgray = imgray;
% results.bw     = bw;

%--------------------------------------------------------------------------