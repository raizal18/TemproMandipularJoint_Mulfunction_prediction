function [genimages]=sequeeze(weights,countOfImage,varargin)
fprintf('\n Generating Image Using EHO-GAN \n')
        for j=1:countOfImage
            genimages(:,:,:,j)=weights(:,:,:,randperm(100,1));
        end
    
fprintf('\n Completed.... \n')
end