%ROI-grid 

im=readimage(imds,5);
%%
figure;
imshow(im)

imr=rgb2gray(im);

%%
pm1=conv2(imr,[16 16],'valid');

pm2=conv2(imr,[32 32],'full');

pm3=conv2(imr,[64 64],'full');

pm4=conv2(imr,[128 128],'full');

figure;imshow([pm1 pm2 pm3 pm4],[])

[m,n]=size(imr);

k=[5 5];

stepSize=10;

