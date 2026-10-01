
clc
close all
%% II
%a
N=20;
n=0:N-1;
x=sin(2*pi*3*n/N) ;
%b
figure
stem(n,x, 'fill');
%c
e=exp(1);
z = 2^(e*i*pi/4 );
y = x*z;
figure
plot(y)
%d
comper=abs(y)./abs(x);
%e
yAngle=angle(y);
xAngle=angle(x);
zAngle=angle(z);


%% 3
%a
%N=randi([40 100]);
N=500
n=0:(N-1);z1=n; z2=n;
for ii=1:length(n)
    k=5;
    z1(ii) = e^(-i*pi*2*k*n(ii)/N );
    k=N-5;
    z2(ii) = e^(-i*pi*2*(n(ii)*k)/N );
end
theta=[angle(z1);angle(z2)];
sinSpace=-5*pi:0.01:5*pi;
figure
subplot(221)
plot(sin(theta(1,:)))
hold on
stem(imag(z1))
title('k=5, imag/sin part of complex number')
legend('sin functin','img. part of vector ')
subplot(222)
hold on
plot(cos(theta(1,:)))
stem(real(z1))
title('k=5, real/cos part of complex number')
legend('cos functin','real part of vector ')
subplot(223)
hold on
plot(cos(theta(2,:)))
stem(real(z2))
title('k=N-5, real/scs part of complex number')
legend('cos functin','real part of vector ')
subplot(224)
plot(sin(theta(2,:)))
title('k=N-5, imag/sin part of complex number')
hold on
stem(imag(z2))
title('k=N-5, imag/sin part of complex number')
legend('sin functin','img. part of vector ')
%% 4
N=16;k=2;n=0:N-1;
h = [round(10*randn(1,N))];
%a
for ii=1:length(h)
    T(:,ii)=circshift(h',ii-1);
end
%b
y1=cos(2*pi*k*n/N);  y2=sin(2*pi*k*n/N);
%c
figure
plot(y1)
hold on
plot(y2)
legend('y1','y2');
%d
s2 = y1'- j*y2';
l=length(s2)
figure
plot(s2(1:(l/2)),'--r')
hold on
plot(s2(l:-1:(l/2+1)),'--')
legend('first cycle', 'second cycle');
%e
z = T*s2;
figure
plot(z(1:(l/2)),'--r')
hold on
plot(z(l:-1:(l/2+1)),'--')
legend('first cycle', 'second cycle');
%f
big=z./s2;
%g
e=eig(T);
%h
%c=dot(h,conj(s2)); used to check

b=conj(s2);
c2=sum(conj(h).*b');