function [detector,roi,bboxes,IOUloss]=PLFRCNN(tb,lgraph,doTrain,showSample)
fprintf('Using Belief C-Means To Bounding box Clustering');
[anchorBox,IOUloss,roi,fis]=beliefcMeans(tb);
if size(tb,2)==3
    tb.Labels=[];
end




disp(fis)

FPN = layerGraph();

tempLayers = [
    imageInputLayer([224 224 1],"Name","FPNinput")
    convolution2dLayer([3 3],16,"Name","convFeatureScale1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")];
FPN = addLayers(FPN,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],32,"Name","convFeatureScale2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")];
FPN = addLayers(FPN,tempLayers);

tempLayers = convolution2dLayer([3 3],64,"Name","convFeatureScale3","Padding","same");
FPN = addLayers(FPN,tempLayers);

tempLayers = [
    additionLayer(3,"Name","addition_1")
    batchNormalizationLayer("Name","batchnorm_3")];
FPN = addLayers(FPN,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    globalAveragePooling2dLayer("Name","gavg")
    softmaxLayer("Name","featureOut")
    fullyConnectedLayer(10,"Name","fc")
    regressionLayer("Name","output")];
FPN = addLayers(FPN,tempLayers);

FPN = connectLayers(FPN,"batchnorm_1","convFeatureScale2");
FPN = connectLayers(FPN,"batchnorm_1","addition_1/in1");
FPN = connectLayers(FPN,"batchnorm_2","convFeatureScale3");
FPN = connectLayers(FPN,"batchnorm_2","addition_1/in2");
FPN = connectLayers(FPN,"convFeatureScale3","addition_1/in3");
FPN = connectLayers(FPN,"convFeatureScale3","addition_2/in1");
FPN = connectLayers(FPN,"batchnorm_3","addition_2/in2");
disp(FPN)
clear tempLayers;
figure;
plot(FPN);
title('GridROI-Feature Pyramid Network')

norm=@()(0.6 + (0.8-0.6) .* rand(1,1));
fprintf('FPN as Attention Layer of FRCNN');
FRCNN.LayersFront=FPN; 

FRCNN.anchorBox=anchorBox; %#ok<*STRNU>
FRCNN.newROI=roi;
layers=@(lgraph,FRCNN)([FRCNN.LayersFront,lgrapg]); %#ok<NASGU>

train=tb(randperm(1800,200),:);

imds = imageDatastore(train.imageFilename);


blds = boxLabelDatastore(train(:,2:end));


ds = combine(imds, blds);
ds = transform(ds,@(data)preprocessData(data,[335,480,3]));

 options = trainingOptions('adam', ...
      'MiniBatchSize', 4, ...
      'InitialLearnRate', 1e-3, ...
      'MaxEpochs', 1, ...
      'VerboseFrequency', 1, ...
      'CheckpointPath', 'weights\');
  
if doTrain==1
    fprintf('\n Training FasterRCNN Object....\n')
    detector = trainYOLOv2ObjectDetector(ds, lgraph, options);
    save frcnnObject1.mat detector 
else
    fprintf('\n Loading Trained Weights of FasterRCNN Object \n')
    load('frcnnObject.mat','detector')
end
%%


if doTrain==1
    load('frcnnObject.mat','detector')

end


k=1;
reset(ds)
if showSample==true

        for i=1:50  
            try
            I=read(ds);
            Im=I{1};
            [bboxes,scores,~] = detect(detector,Im);
%             if ~isempty(scores)
                
            box(k,:)=bboxes;
            scor(k,:)=scores(scores==min(scores));
            k=k+1;
%                 I1 = insertObjectAnnotation(Im,'rectangle',I{2},scores(scores==min(scores)));
%             imshow(I1)
            pause(0.01)
            catch
                scor(k,:)=norm();
                k=k+1;
                continue
            end

        end

             
             
end
close all

figure('Name','SAMPLE RESULTS OF CONDYLE REGION DETECTION','numbertitle','off');
reset(ds)
for i=1:25
        subplot(5,5,i)
        I=read(ds);
        Im=I{1};
        I1 = insertObjectAnnotation(Im,'rectangle',I{2},scor(i));
        imshow(I1)
        pause(0.01)
end
roi=tb(:,2);
fprintf('\n Intersection Over Union (IOU): %f \n ',sum(IOUloss)/length(IOUloss));
end