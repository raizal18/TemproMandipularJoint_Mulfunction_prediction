function values=roi2feature(varargin)
Data=varargin{1};
for i=1:length(Data)
values(i,:)=Data{i};
end
end