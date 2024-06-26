clc;
clear;
close all;
filepath='image3D';

im=imageDatastore(filepath,...
    'includesubfolders',true,...
    'LabelSource','foldernames');

count=countEachLabel(im);
fprintf('\n Total TMJ image Counts :\n')
disp(count)
[train,test]=splitEachLabel(im,0.7);
count=countEachLabel(train);
fprintf('\n Train Size :\n')
disp(count)
count=countEachLabel(test);
fprintf('\n Test Size :\n')
disp(count)
figure('name','Input images');
for i=1:25
    subplot(5,5,i)
    imshow(read(test))
end
info=struct;
%%
close;
s=@(v)(v(:,length(pwd)+2:end));
show=@(p)(imshow(imread(p)));
% [I,rec]=imcrop(imread(im.Files{159}));

for i=1:length(im.Files)
    val{i,:}=s(im.Files{i});
end
    
info.imageFilename=val;
info.Labels=im.Labels;
Condyle(1:158,:)  ={[158.5 47.5 187 219]};
Condyle(159:297,:)={[76.5 53.5 165 222]};
Condyle(298:391,:)={[188.5 36.5 198 231]};
Condyle(392:470,:)={[183.5 81.5 164 215]};
Condyle(471:544,:)={[147.5 67.5 160 200]};
Condyle(545:623,:)={[130.5 19.5 158 199]};
Condyle(624:698,:)={[209.5 186.5 121 149]};
Condyle(689:758,:)={[207.5 152.5 140 183]};
Condyle(759:828,:)={[114.5 109.5 153 200]};
Condyle(829:889,:)={[114.5 109.5 153 200]};
Condyle(890:1009,:)={[106.5 57.5 168 222]};
Condyle(1010:1092,:)={[108.5 67.5 152 196]};
Condyle(1093:1162,:)={[73.5 96.5 151 211]};
Condyle(1163:1237,:)={[226.5 159.5 179 176]};
Condyle(1238:1312,:)={[182.5 131.5 150 204]};
Condyle(1313:1389,:)={[46.5 114.5 148 221]};
Condyle(1390:1463,:)={[69.5 93.5 155 233]};
Condyle(1464:1538,:)={[35.5 69.5 160 214]};
Condyle(1539:1613,:)={[51.5 39.5 148 216]};
Condyle(1614:1692,:)={[103.5 135.5 143 200]};
Condyle(1693:1779,:)={[84.5 106.5 148 201]};
Condyle(1780:1831,:)={[126.5 92.5 175 237]};
Condyle(1832:1887,:)={[142.5 84.5 134 204]};



info.Condyle=Condyle;
tb=struct2table(info);



a=imageDatastore('ROI');


imageData=a.Files;

imageClass=tb.Labels;

clear train test im a img blds info


%% Generating Image Using EHO-GAN 

numLatentInputs=100;

executionEnvironment = "auto";

[Generator,discriminator]=EHOGAN(imageData,imageClass,numLatentInputs);



ZNew = randn(1,1,numLatentInputs,16,'single');
dlZNew = dlarray(ZNew,'SSCB');


if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZNew = gpuArray(dlZNew);
end

%%
countOfImage=100;
load('weights.mat','weights')

[genimages]=sequeeze(weights,countOfImage,Generator);
figure;


I=imtile(genimages(:,:,:,1:100));
I=rescale(I);
image(I)
title(...
                "Epoch: " + 500 + ", " + ...
                "Iteration: " + 10000)

drawnow;




%%
blds = boxLabelDatastore(tb(:,3:end));


numAnchors = 5;
anchorBoxes = estimateAnchorBoxes(blds,numAnchors);

inputImageSize = [335,480,3];

numClasses = 1;

network = alexnet();
featureLayer = 'relu5';
lgraph = yolov2Layers(inputImageSize,numClasses,anchorBoxes,network, featureLayer);

% set below flag to 1 for Traing frcnn from sratch

trainFromSratch=0;
showSample=true;
[frcnndetector,roi,bboxes,iou]=PLFRCNN(tb,lgraph,trainFromSratch,showSample);

% analyzeNetwork(lgraph)

fprintf('\n ROI Extraction in Progress \n')
for i=1:size(tb,1)

    img=imread(tb.imageFilename{i});
    r=roi.Condyle{i};
    img=imcrop(img,r);
    imwrite(imresize(img,[224,224]),['ROI\',sprintf('im%04d.bmp',i)])

end
fprintf('\n ROI Extraction Completed \n')

%%

for i=1:size(genimages,4)
imwrite(genimages(:,:,:,i),['generatedImage/OSTEOARTHRITIS/',sprintf('gimg %03d.bmp',i)])
end

newData=imageDatastore('generatedImage','IncludeSubfolders',true,'LabelSource','folderNames');


imds=imageDatastore(imageData,'Labels',imageClass);

newFiles=[imds.Files;newData.Files];
newLabels=[imds.Labels;newData.Labels];

imds.Files=newFiles;
imds.Labels=newLabels;

fprintf('Image Size after EHOGAN \n')
disp(countEachLabel(imds))

% set below Flag if Want To Extract All Dataimages In Database
ExtractForAll=false;


[c_matrix1,Result1,predict1,actual1]=classifier(imds, ExtractForAll);

load('m0.mat')
load('m1.mat')
load('m2.mat')

acc3=Result1.Accuracy;
pre3=Result1.Precision;
re3=Result1.Sensitivity;
f3=Result1.F1_score;
y=[acc3,pre3,re3,f3];
%
save m3.mat acc3 pre3 re3 f3

figure;

heatmap(c_matrix1);
xlabel('Actual class')
ylabel('Predicted class')

%%
s=1;
titleshow=1;
legendpos=1;
Y=[acc0*100; acc1*100; acc2*100; acc3*100];
algn = {'FRCNN','EHOGAN-FRCNN','EHOGAN-PFRCNN','EHOGAN-PLFRCNN'};
titlen='Accuracy';
xl='Model';
yl='Accuracy (%)';
xval='';
n=size(Y,3);
xa=[1,n+2];
ya=[50,100];
saveAsName='Accuracy';
graph.bar(Y,algn,titlen,xl,yl,xval,xa,ya,titleshow,legendpos,s,saveAsName)
%%
s=1;
titleshow=1;
legendpos=1;
Y=[pre0*100; pre1*100; pre2*100; pre3*100];
algn = {'FRCNN','EHOGAN-FRCNN','EHOGAN-PFRCNN','EHOGAN-PLFRCNN'};
titlen='Precision';
xl='Model';
yl='Precision (%)';
xval='';
n=size(Y,3);
xa=[1,n+2];
ya=[50,100];
saveAsName='Precision';
graph.bar(Y,algn,titlen,xl,yl,xval,xa,ya,titleshow,legendpos,s,saveAsName)
%%
s=1;
titleshow=1;
legendpos=1;
Y=[re0*100; re1*100; re2*100; re3*100];
algn = {'FRCNN','EHOGAN-FRCNN','EHOGAN-PFRCNN','EHOGAN-PLFRCNN'};
titlen='Recall';
xl='Model';
yl='Recall (%)';
xval='';
n=size(Y,3);
xa=[1,n+2];
ya=[50,100];
saveAsName='Recall';
graph.bar(Y,algn,titlen,xl,yl,xval,xa,ya,titleshow,legendpos,s,saveAsName)
%%
s=1;
titleshow=1;
legendpos=1;
Y=[f0*100; f1*100; f2*100; f3*100];
algn = {'FRCNN','EHOGAN-FRCNN','EHOGAN-PFRCNN','EHOGAN-PLFRCNN'};
titlen='F1-Score';
xl='Model';
yl='F1-Score (%)';
xval='';
n=size(Y,3);
xa=[1,n+2];
ya=[50,100];
saveAsName='F1Score';
graph.bar(Y,algn,titlen,xl,yl,xval,xa,ya,titleshow,legendpos,s,saveAsName)

fprintf('\n IOU : %f \n',(sum(iou)/length(iou)))



