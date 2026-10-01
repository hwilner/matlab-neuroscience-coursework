clear all
close all
clc

%q1
%just for checking my results
 z = 0.9*exp(i*pi/3);
 a=angle(z);
 amp=abs(z);
 %figure 
% plot(z^2)

%q2
N=20;
n=0:N-1;
x=sin((2*pi*3)*n/N);
figure
stem(n,x, 'fill')
z1 = 2*exp((i*pi)/4);
y = x*z1; 
figure
plot (y) 

compare=abs(y)./abs(x);
angy=angle(y);
angx=angle(x);
angz=angle(z1);
figure
plot(angy)
hold on
plot(angx,'r')
hold on
plot(angz,'g')

%q3
N=randi([40,100]);
n=0:N-1;

k=5;
cmx=exp((-j*2*pi*k)*(n/N));
figure
subplot(2,3,1)
plot(cmx)
title('k=5, plotting of complex number')
hold on
subplot(2,3,2)
plot(real(cmx))
title('k=5, plotting of real part of complex number')
hold on
subplot(2,3,3)
plot(imag(cmx))
title('k=5, plotting of img part of complex number')

%N=75;
%n=0:N-1;
k=N-5;
cmx2=exp((-j*2*pi*k)*(n/N));
hold on
subplot(2,3,4)
plot(cmx2,'r')
title('k=N-5, plotting of complex number')
hold on
subplot(2,3,5)
plot(real(cmx2),'r')
title('k=N-5, plotting of real part of complex number')
hold on
subplot(2,3,6)
plot(imag(cmx2),'r')
title('k=N-5, plotting of img part of complex number')

%q4
N=16;
k=2;
n=0:N-1;
h =[round(10*randn(1,N))];
h=h';

T=zeros(16,16);

ytag=zeros(1,16);
ytag=ytag';
for ii=0:15;
    ytag=zeros(1,16);
    ytag=circshift(h,ii);
    T(:,ii+1)=ytag;
end
figure
imagesc(T)

y1=cos(2*pi*k*(n/N)); 
y2=sin(2*pi*k*(n/N)); 

%plotting

subplot(1,3,1)
plot (y1)
title('Y1')
hold on
subplot(1,3,2)
plot(y2,'r')
title('Y2')
subplot(1,3,3)
hold on
plot(y1)
hold on
plot(y2,'r')
title('Y1 & Y2')

s2 = y1- j*y2;
figure
plot([1:N],s2)
zT= T*s2' ;
% how big is z reltaive to s2?
%relative=zT./s2';
figure
plot([1:N],zT,'r')

compz=(zT)./(s2');

e=eig(T);

%<h,conj(s2)> 

conjS2=conj(s2);
%check=(h)*conjS2;
check=h'*conjS2';

%convolution
%2 vectors
hvector=T(:,1);
s2 = y1- j*y2;

fliped_s2=fliplr(s2);
zconv=conv(hvector,fliped_s2);



