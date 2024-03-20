out='image3D/';
for i=1:size(tb,1)
    if strcmp(string(tb{i,2}),"NORMAL")
        p='image3D\NORMAL\';
    else
        p='image3D\OSTEOARTHRITIS\';
    end
        name=tb{i,1};
    img=imread(name{1});
    nimg=cat(3,img,img,img);
    imwrite(nimg,[p,sprintf('image %04d.bmp',i)])
end
