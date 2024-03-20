V=V4;
for i=1:size(V,4)
disp(i)
I=V(:,:,:,i);
I=rescale(I);
imwrite(I,['KARTHIK\VOL_4\',sprintf('%03d',i),'.png'])
end
%%
V=V5;
for i=1:size(V,4)
disp(i)
I=V(:,:,:,i);
I=rescale(I);
imwrite(I,['KARTHIK\VOL_5\',sprintf('%03d',i),'.png'])
end
%%
V=V6;
for i=1:size(V,4)
disp(i)
I=V(:,:,:,i);
I=rescale(I);
imwrite(I,['HARISHTHA\VOL_2\',sprintf('%03d',i),'.png'])
end
%%
V=V7;
for i=1:size(V,4)
disp(i)
I=V(:,:,:,i);
I=rescale(I);
imwrite(I,['HARISHTHA\VOL_3\',sprintf('%03d',i),'.png'])
end







%%
for i=1:17
    clear(sprintf('spatialDetails%d',i))
end
%%
for i=1:17
    clear(sprintf('V%d',i))
end

