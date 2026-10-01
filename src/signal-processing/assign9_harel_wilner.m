close all;
%% 2
x=rand(1,20);
X=fft(x); % DFT of x
y = circshift(x,4);% Circular shift the signal x by 4 points
Y=fft(y); % DFT of y
m=4;k=x;N=20; e=exp(-j*2*pi*k(:)*m/N);
XY=e'+X
ver=(XY==y);
%% 3
%Construct two random signals of length n
n=300;
sig1=rand(1,n);
sig2=rand(1,n);
%cyclic convolution between the two signals
ccirc = ifft(fft(sig1).*fft(sig2));
%d:
milti=fft(sig1,length(ccirc)).*fft(sig2,length(ccirc));
cirdft=fft(ccirc,length(ccirc));
figure
subplot(211)
stem(milti)
title('multiplication of the two signal DFT')
subplot(212)

stem(cirdft)
title('DFT of the twosignals cyclic conv')

%% 4
N=100;
n=1:N;
x=3*sin(2*pi*2*n/N)+sin(2*pi*32*n/N);
X=(fft(x));

%a:
H= [0.2*ones(1,5) ,  zeros(1,length(x)-5)];
h=ifft(H);
figure
subplot(231)
stem(h); title('h')
subplot(234)
stem(abs(fft(h)));title('H');
subplot(232)
stem(x);title('x');
subplot(235)
stem(abs(fft(x)));title('X');
subplot(233)
stem((conv(fftshift(h),x)));title('conv(fftshift(h),x)');
subplot(236)
stem(ifft(H.*X));title('ifft(HX)');
%b:
H= [  zeros(1,length(x)-10) ,[0.5 -1 1 -2 2 2 -2 1 -1 0.5]];
h=ifft(H);
figure
subplot(231)
stem(h); title('h')
subplot(234)
stem(abs(fft(h)));title('H');
subplot(232)
stem(x);title('x');
subplot(235)
stem(abs(fft(x)));title('X');
subplot(233)
stem((conv(fftshift(h),x)));title('conv(fftshift(h),x)');
subplot(236)
stem(ifft(H.*X));title('ifft(HX)');

%% 5
%two sinus signals:
fs=1000;
t=0:1/fs:1-1/fs;
sinSig1=sin(2*pi*t*2); %2hz
sinSig2=sin(2*pi*t*5); %5hz
h=[1 2 3 2 1 zeros(1,length(t)-5)];
% a:
figure
subplot(231)
stem(sinSig1);title('2 Hz sin wave')
subplot(232)
stem(sinSig2);title('5 Hz sin wave')
subplot(234)
stem(h);title('h')
subplot(235)
stem(abs(fft(h)));title('abs(H)')
subplot(236)
stem(angle(fft(h)));title('angle H')

%b:
h=[1 2 3 2 1 zeros(1,length(t)-5)];
newSin1=ifft(fft(sinSig1)'*h);
newSin2=ifft(fft(sinSig2)'*h);
%c:
phaseshift=max(angle(sinSig1))-max(angle(newSin1));
ver=(phaseshift==(max(angle(sinSig2))-max(angle(newSin2))));

%d:
figure
subplot(211)
stem(sinSig1);title('2 Hz sin wave')
hold on
plot(newSin1,'--m')
subplot(212)
stem(sinSig2);title('5 Hz sin wave')
hold on
plot(newSin2,'--m')
%% 6
y=[1 2 3 4 1 2 3 4 8 9 1 2];
h=[0.2 0.4 1 0.4 0.2];
x=deconv(y,h)
ver= (y==conv(x,h))



