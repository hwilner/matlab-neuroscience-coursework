close all
%% Question 2: DFT of important functions
   Fs = 60;                   % samples per second
   dt = 1/Fs;                   % seconds per sample
   StopTime = 0.5;             % seconds
   tw = (0:dt:StopTime-dt)';     % seconds
                  
% 1
fc=10 ;
x=sin(2*pi*fc*tw);
N = length(x);
X = zeros(N,1);
for k = 0:N-1
    for n = 0:N-1
        X(k+1) = X(k+1) + x(n+1)*exp(-j*pi/2*n*k);
    end
end
figure
t = 0:N-1
subplot(311)
stem(t,x);
xlabel('Time (s)');
ylabel('Amplitude');
title('Time domain - Input sequence')

subplot(312)
stem(t,X)
xlabel('Frequency');
ylabel('|X(k)|');
title('Frequency domain - Magnitude response')

subplot(313)
stem(t,angle(X))
xlabel('Frequency');
ylabel('Phase');
title('Frequency domain - Phase response')

% 2
fc=5 ;
x=cos(2*pi*fc*tw);
N = length(x);
X = zeros(N,1);
for k = 0:N-1
    for n = 0:N-1
        X(k+1) = X(k+1) + x(n+1)*exp(-1i*pi/2*n*k);
    end
end
figure
t = 0:N-1
subplot(311)
stem(t,x);
xlabel('Time (s)');
ylabel('Amplitude');
title('Time domain - Input sequence')

subplot(312)
stem(t,X)
xlabel('Frequency');
ylabel('|X(k)|');
title('Frequency domain - Magnitude response')

subplot(313)
stem(t,angle(X))
xlabel('Frequency');
ylabel('Phase');
title('Frequency domain - Phase response')

% 3
x=[ zeros(1,20) ones(1,5) zeros(1,19)];
N = length(x);
X = zeros(N,1);
for k = 0:N-1
    for n = 0:N-1
        X(k+1) = X(k+1) + x(n+1)*exp(-1i*pi/2*n*k);
    end
end
figure
t = 0:N-1
subplot(311)
stem(t,x);
xlabel('Time (s)');
ylabel('Amplitude');
title('Time domain - Input sequence')

subplot(312)
stem(t,X)
xlabel('Frequency');
ylabel('|X(k)|');
title('Frequency domain - Magnitude response')

subplot(313)
stem(t,angle(X))
xlabel('Frequency');
ylabel('Phase');
title('Frequency domain - Phase response')

%4
x=[ zeros(1,20) ones(1,5) zeros(1,19)];
N = length(x);
X = zeros(N,1);
for n = 0:N-1
    for k = 0:N-1
        X(k+1) = X(n+1) + x(k+1)*exp(2*i*pi*n*k/N);
    end
end
figure
t = 0:N-1
subplot(311)
stem(t,x);
xlabel('Time (s)');
ylabel('Amplitude');
title('Time domain - Input sequence')

subplot(312)
stem(t,X)
xlabel('Frequency');
ylabel('|X(k)|');
title('Frequency domain - Magnitude response')

subplot(313)
stem(t,angle(X))
xlabel('Frequency');
ylabel('Phase');
title('Frequency domain - Phase response')

%% Question 3: Modulation in time and frequency
%3&4
x=[0 0 0 1 2 3 4 3 2 1 0 0];
K=length(x)-2;
y = circshift(x,K);
N = length(x);
X = zeros(N,1);
Y = zeros(N,1);
for k = 0:N-1
    for n = 0:N-1
        X(k+1) = X(k+1) + x(n+1)*exp(-1i*pi/2*n*k);
        Y(k+1) = Y(k+1) + y(n+1)*exp(-1i*pi/2*n*k);
    end
end
figure
t = 0:N-1;
subplot(121)
stem(t,x);
xlabel('Time (s)');
ylabel('Amplitude');
title('Time domain - x')
subplot(122)
stem(t,y);
xlabel('Time (s)');
ylabel('Amplitude');
title('Time domain - shift x')

%5

fft_a = fft(X); 
fft_a = fft_a(2:end); %drop the first point
angle_a = angle(fft_a); %angle() the result
fft_b = fft(Y); 
fft_b = fft_b(2:end); %drop the first point
angle_b = angle(fft_b); %angle() the result
phase_shift = abs(angle_a - angle_b);
ver=phase_shift;
e=exp(1);
for m=1:(N-1)
ver = e^(2*pi*K*m/N);
end

%% Question 4:  DFT in action
[x,fs]=auread('bluewhale.au');

figure
plot(x)
xlabel('Sample Number')
ylabel('Amplitude')
title('Blue Whale Call')
ind1=23912; ind2=37204; %chosen manually

moan=x(ind1:ind2);  
N = length(moan);
X = zeros(N,1);
for n = 0:N-1
    for k = 0:N-1
        X(k+1) = X(n+1) + moan(k+1)*exp(2*i*pi*n*k/N);
    end
end
figure
t = 0:N-1
subplot(311)
plot(t,moan);
xlabel('Time (s)');
ylabel('Amplitude');
title('Time domain - Input sequence')

subplot(312)
plot(t,X)
xlabel('Frequency');
ylabel('|X(k)|');
title('Frequency domain - Magnitude response')

subplot(313)
plot(t,angle(X))
xlabel('Frequency');
ylabel('Phase');
title('Frequency domain - Phase response')

trill=x((0.6*10^4):(1.3*10^4));
trill_dft=fft(trill);
N=length(trill);


[S,F,T]=spectrogram(y, 128, 120, 256, Fs);
spectrogram(y, 128, 120, 256, Fs)
abs_trill_vec=abs(trill_dft);
max_val_trill=max(abs_trill_vec);
ind_max_trill=find(abs_trill_vec==max_val_trill);
freq_trill=4000/N;


%%Q5

%q5
figure
plot(t,d)
a=fft(d);
N=length(d);
figure
plot(abs(a(1:(N/2))))
set(gca,'YLim',[0 2000])
set(gca,'XLim',[0 2000])

abs_vec=abs(a);
sorted_a=sort(abs_vec);
max_val=sorted_a(299999:300000);
max_ind=find(abs_vec==max_val(1));
max_ind(3)=find(abs_vec==max_val(2));
%fs=30,000 hz, t=10 sec
freq=30000/10;

%calculate the max val. then find, and calc the freq jumps

low_pass300=a;
low_pass300(1:3000)=[];
inverse_low_pass=ifft(low_pass300);
N=length(inverse_low_pass);
figure
plot(abs(inverse_low_pass(1:(N/2))))
d2 = d-(-0.1);%d2 is the same length of d
 
S = [0 d2(1:end-1).*d2(2:end)];
indx = find(S < 0 & d2 < 0 );
wv=[];
for i=1:length(indx)
    wv(end+1,:) = d2(indx(i)-26:indx(i)+24);
end

%r=wv';
figure
plot(wv')

%% Question 6: Convolution
x = round(rand(1,30));
%a
y= [0.2 0.2 0.2 0.2 0.2 zeros(1,25)];
y=(smooth(x,5))';
h=zeros(length(y),length(y));
for ii=1:length(y)
   h(:,ii)=circshift(y',ii-1);  
end
h2=x*h/5;
h3=h*h;
%b
c=[zeros(1,26) 1  0 0 0];
for ii=1:length(c)
   h(:,ii)=circshift(c',ii-1); 
end
c2=x*h;

%c


%d
d=smooth(x,10);
d2=smooth(smooth(x,5),5) ;




