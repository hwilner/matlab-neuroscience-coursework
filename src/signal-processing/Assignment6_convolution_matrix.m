function Assignment6_convolution_matrix

Delta=[1 0 0 0 0 0 0 0 0 0];
y= [1 2 1 0 0 0 0 0 0 0];
h=zeros(length(y),length(y))
%% b
%matrix H
for ii=1:length(y)
   h(:,ii)=circshift(y',ii-1)
   
end
figure
imagesc(h)
colormap(flipud(copper))

%% f
x=[0 -1 -2 3 4 3 4 1 2 0 ]';
ansD=h*x

%% g
ansG=cconv(x,h(:,2),10)

end