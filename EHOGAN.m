function [dlnetGenerator,dlnetDiscriminator]=EHOGAN(varargin)
%% EHO-GAN

a=imageDatastore('ROI');

load('roiLabels.mat','Labels')

imageData=a.Files;

imageClass=Labels;

clear a

file=imageData(imageClass==categorical({'OSTEOARTHRITIS'}));
label=imageClass(imageClass==categorical({'OSTEOARTHRITIS'}));
imds=imageDatastore(file,'Labels',label);



[~,train]=splitEachLabel(imds,0.6);

% test.ReadFcn=@customReadDatastoreImage;
% train.ReadFcn=@customReadDatastoreImage;

augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandScale',[1 2]);
augimds = augmentedImageDatastore([64,64],train,'DataAugmentation',augmenter);




numEpochs = 3;
miniBatchSize = 20;

augimds.MiniBatchSize=miniBatchSize;

%%
filterSize = [4 4];
numFilters = 64;
numLatentInputs = 100;
layersGenerator = [
    imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','in')
    transposedConv2dLayer(filterSize,8*numFilters,'Name','tconv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    transposedConv2dLayer(filterSize,4*numFilters,'Stride',2,'Cropping',1,'Name','tconv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    transposedConv2dLayer(filterSize,2*numFilters,'Stride',2,'Cropping',1,'Name','tconv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    transposedConv2dLayer(filterSize,numFilters,'Stride',2,'Cropping',1,'Name','tconv4')
    batchNormalizationLayer('Name','bn4')
    reluLayer('Name','relu4')
    transposedConv2dLayer(filterSize,3,'Stride',2,'Cropping',1,'Name','tconv5')
    tanhLayer('Name','tanh')];

lgraphGenerator = layerGraph(layersGenerator);


dlnetGenerator = dlnetwork(lgraphGenerator);

scale = 0.2;

layersDiscriminator = [
    imageInputLayer([64 64 3],'Normalization','none','Name','in')
    convolution2dLayer(filterSize,numFilters,'Stride',2,'Padding',1,'Name','conv1')
    leakyReluLayer(scale,'Name','lrelu1')
    convolution2dLayer(filterSize,2*numFilters,'Stride',2,'Padding',1,'Name','conv2')
    batchNormalizationLayer('Name','bn2')
    leakyReluLayer(scale,'Name','lrelu2')
    convolution2dLayer(filterSize,4*numFilters,'Stride',2,'Padding',1,'Name','conv3')
    batchNormalizationLayer('Name','bn3')
    leakyReluLayer(scale,'Name','lrelu3')
    convolution2dLayer(filterSize,8*numFilters,'Stride',2,'Padding',1,'Name','conv4')
    batchNormalizationLayer('Name','bn4')
    leakyReluLayer(scale,'Name','lrelu4')
    convolution2dLayer(filterSize,1,'Name','conv5')];

lgraphDiscriminator = layerGraph(layersDiscriminator);


dlnetDiscriminator = dlnetwork(lgraphDiscriminator)




figure
subplot(1,2,1)
plot(lgraphGenerator)
title("Generator")

subplot(1,2,2)
plot(lgraphDiscriminator)
title("Discriminator")


%% EHO Parameter initiating


learnRateGenerator = 0.0002;
learnRateDiscriminator = 0.0001;

trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];

gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;



executionEnvironment = "auto";

ZValidation = randn(1,1,numLatentInputs,64,'single');
dlZValidation = dlarray(ZValidation,'SSCB');

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZValidation = gpuArray(dlZValidation);
end

%%

figure
iteration = 0;
start = tic;

% Loop over epochs.
for i = 1:numEpochs
    
    % Reset and shuffle datastore.
    reset(augimds);
    augimds = shuffle(augimds);
    
    
    while hasdata(augimds)
        iteration = iteration + 1;
        fprintf('\n Optimization Using Herding  Iter %d \n',iteration);
        
        data = read(augimds);
        
        
        if size(data,1) < miniBatchSize
            continue
        end
        
        
        
        X = cat(4,data{:,1}{:});
        Z = randn(1,1,numLatentInputs,size(X,4),'single');
        
        % Normalize the images
        X = (single(X)/255)*2 - 1;
        
        % Convert mini-batch of data to dlarray specify the dimension labels
        % 'SSCB' (spatial, spatial, channel, batch).
        dlX = dlarray(X, 'SSCB');
        dlZ = dlarray(Z, 'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
            dlZ = gpuArray(dlZ);
        end
        
        % Ranking Elephant Using Fitness Fcn
        % dlfeval and the modelGradients function listed at the end of the
        
        [gradientsGenerator, gradientsDiscriminator, stateGenerator] = ...
            dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlX, dlZ);
        dlnetGenerator.State = stateGenerator;
        
        % Update the discriminator network parameters using Clan operator
        [dlnetDiscriminator.Learnables,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator.Learnables, gradientsDiscriminator, ...
            trailingAvgDiscriminator, trailingAvgSqDiscriminator, iteration, ...
            learnRateDiscriminator, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Evaluate the population according to the newly updated locations
        % and replace worst Elephant
        
        [dlnetGenerator.Learnables,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator.Learnables, gradientsGenerator, ...
            trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
            learnRateGenerator, gradientDecayFactor, squaredGradientDecayFactor);
        
        
        % held-out generator input.
        if mod(iteration,10) == 0 || iteration == 1
            
            
            dlXGeneratedValidation = predict(dlnetGenerator,dlZValidation);
            
            
            I = imtile(extractdata(dlXGeneratedValidation));
            I = rescale(I);
            image(I)
            
            
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title(...
                "Epoch: " + i + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D))
            
            drawnow
        end
    end
end

%%
load('ehogan.mat','dlnetDiscriminator','dlnetGenerator')
% ZNew = randn(1,1,numLatentInputs,16,'single');
% dlZNew = dlarray(ZNew,'SSCB');
% 
% 
% if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
%     dlZNew = gpuArray(dlZNew);
% end
% 
% dlXGeneratedNew = predict(dlnetGenerator,dlZNew);
% 
% 
% I = imtile(extractdata(dlXGeneratedNew));
% I = rescale(I);
% image(I)
% title("Generated Images")
fprintf('\n **************Training And Optimization of EHOGAN Completed ************* \n');
end

