function result = batchbatchpro(inDir, outDir)

if(nargin<2)
    outDir = '';
end
if(nargin<1)
    inDir = 'G:\TMJ\TMJ NORMAL\2';
end

includeSubdirectories = true;


workSpaceFields = {
    
};


fileFieldsAndFormat = {
    {'imgray', 'bmp'}
    };



imreadFormats       = imformats;
supportedExtensions = [imreadFormats.ext];

supportedExtensions{end+1} = 'dcm';
supportedExtensions{end+1} = 'ima';
supportedExtensions = strcat('.',supportedExtensions);

supportedExtensions{end+1} = '';



imds = datastore(inDir,...
    'IncludeSubfolders', includeSubdirectories,...
    'Type','image',...
    'FileExtensions',supportedExtensions);
imds.ReadFcn = @readSupportedImage;



result(numel(imds.Files)) = struct();

for ind =1:numel(workSpaceFields)
    [result.(workSpaceFields{ind})] = deal([]);
end



for imgInd = 1:numel(imds.Files)
    
    inImageFile  = imds.Files{imgInd};
    

    outImageFileWithExtension = strrep(inImageFile, inDir, outDir);

    [path, filename,~] = fileparts(outImageFileWithExtension);
    outImageFile = fullfile(path,filename);
    
    try

        im = imds.readimage(imgInd);
        

        oneResult = batchpro(im);
        

        for ind = 1:numel(workSpaceFields)

            fieldName = workSpaceFields{ind};
            result(imgInd).(fieldName) = oneResult.(fieldName);
        end
        

        result(imgInd).fileName = imds.Files{imgInd};
        


        if(~isempty(outDir))

            outSubDir = fileparts(outImageFile);
            createDirectory(outSubDir);
            
            for ind = 1:numel(fileFieldsAndFormat)
                fieldName  = fileFieldsAndFormat{ind}{1};
                fileFormat = fileFieldsAndFormat{ind}{2};
                imageData  = oneResult.(fieldName);

                outImageFileWithExtension = [outImageFile,'_',fieldName, '.', fileFormat];
                
                try
                    imwrite(imageData, outImageFileWithExtension);
                catch IMWRITEFAIL
                    disp(['WRITE FAILED:', inImageFile]);
                    warning(IMWRITEFAIL.identifier, IMWRITEFAIL.message);
                end
            end
        end
        
        disp(['PASSED:', inImageFile]);
        
    catch READANDPROCESSEXCEPTION
        disp(['FAILED:', inImageFile]);
        warning(READANDPROCESSEXCEPTION.identifier, READANDPROCESSEXCEPTION.message);
    end
    
end



end


function img = readSupportedImage(imgFile)

if(isdicom(imgFile))
    img = dicomread(imgFile);
else
    img = imread(imgFile);
end
end

function createDirectory(dirname)
% Make output (sub) directory if needed
if exist(dirname, 'dir')
    return;
end
[success, message] = mkdir(dirname);
if ~success
    disp(['FAILED TO CREATE:', dirname]);
    disp(message);
end
end
