path1='TMJ NORMAL';
path2='TMJ OA';
im1=imageDatastore(path1,'Includesubfolders',true,'LabelSource','folderNames');
im2=imageDatastore(path2,'Includesubfolders',true,'LabelSource','folderNames');