function [c_matrix,Result,predict,actual]=classifier(imds,ExtractForAll)
% [train,test]=splitEachLabel(imds,0.6);

fprintf('\n Load Pretrained Models \n')

net=resnet101();
load('vgg16net.mat','vgg')


vgg16=vgg;

disp(net)
figure('Name','Resnet 101 Structure ','numbertitle','off')
plot(net)


if ExtractForAll==true
imds.reset()

featureExtract=struct;


for i=1:length(imds.Files)
    
    fprintf('Feature Extraction Iteration %03d/%d \n', i,length(imds.Files))
    
   
    
    im=imread(imds.Files{i});    
    
    feat=squeeze(activations(vgg16,im,'fc1000'));
    
    featureExtract.vggfeature(i,:)=feat;
    
    featureExtract.label(i,:)=imds.Labels(i);
    
    featureExtract.filename(i,:)=imds.Files{i};
end
else
  imds.reset()

featureExtract=struct;


for i=1:100
    
    fprintf('Feature Extraction Iteration %03d/%d \n', i,100)
    
 
    
    im=imread(imds.Files{i});    
    
    feat=squeeze(activations(vgg16,im,'fc1000'));
    
    featureExtract.vggfeature(i,:)=feat;
    
    featureExtract.label(i,:)=imds.Labels(i);
    
    featureExtract.filename(i,:)=imds.Files{i};
end  
 load('feature.mat','featureExtract')
end


X=featureExtract.vggfeature;
Y=featureExtract.label;

cv=cvpartition(Y,'holdOut',0.7);
xtrain=X(~cv.test,:);
xtest=X(cv.test,:);
ytrain=Y(~cv.test);
ytest=Y(cv.test);

mdl=fitcecoc(xtrain,ytrain);

save classifer1.mat mdl
ypred=mdl.predict(xtest);



met1=confusion1;

[c_matrix,Result,~,predict,actual]=met1.getMatrix(double(ytest),double(ypred),8);


% heatmap(c_matrix)
% xlabel('Actual class')
% ylabel('Predicted class')
% 
% rs=struct2array(Result);
% b=bar(rs([1,3,5,7]),0.25);


end