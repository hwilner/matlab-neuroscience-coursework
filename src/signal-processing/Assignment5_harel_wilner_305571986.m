function  Assignment5_harel_wilner_305571986

%% Q2
x=[0 0 1 2 1 0 0 ]; 
y=ones(1,7);
n1=length(x);
n2 = length(y);
n = max(n1,n2);
t=1:n;
x = [x,zeros(1,n-n1)];
y = [y,zeros(1,n-n2)];
yh = zeros(1,n);
for i =0:n-1
            for j = 0:n-1
                      k = mod((i-j),n);
                      yh(i+1) = yh(i+1) + x(j+1)*y(k+1);
            end
end
figure
subplot(2,1,1)
stem(t,yh,'c','MarkerEdgeColor','b')
title('my circular convolution ');
%ym=cconv(circshift(fliplr(y),n), circshift(fliplr(x),n),n)




subplot(2,1,2)
check=cconv(x,y,length(x));
stem(t,check,'b','MarkerEdgeColor','c')
title('matlab circular convolution ');
%%

H=[1 2 3; 3 1 2; 2 3 1];
x=[5 6 7]';
ans1=H*x;
x21=circshift(x,1)
end
