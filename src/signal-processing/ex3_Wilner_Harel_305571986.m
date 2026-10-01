function  ex3_Wilner_Harel_305571986
%%%Assignment #3/
close all

%% Exercise 1: SNR
y=randn(1,400)/10+ [zeros(1,100) ones(1,200) zeros(1,100)]+5;

%1.Normalized data with mean 0:
% the most simple option is to normalize it to z values: Z = zscore(y),
%but its demends the 'stats toolbox', so we do it manually:
%subtracte the mean from y:
x=(y(101:200)-mean(y(101:200)));
%re-arange to z scale
z=x/std(x(:));

%2.	Computing the noise and signal power:
noise=y-[zeros(1,100) ones(1,200) zeros(1,100)]+5;
Snoise = std(noise) %noise powe
Ssignal = mean(y(101:200)) %signal power


%3.	Computing the SNRpower , SNRrms and db values.
SNRp = Ssignal.^2/Snoise^2 %SNR power
SNRrms=SNRp.^0.5 %SNR RMS
db= 10*log10(SNRp) 

%4.
y2=randn(1,400)/2+ [zeros(1,100) ones(1,200) zeros(1,100)]+5;
x2=(y2)-mean(y2);
z2=x2/std(x2);
noise=y2-[zeros(1,100) ones(1,200) zeros(1,100)]+5;

signalP2=mean(y2(101:200))
sNoise2=std(noise)
snrPower2=signalP2/sNoise2
snrRMS2=snrPower2.^0.5

figure

plot (y2,'color', 'b');
xlabel('time (sec)');
ylabel(' voltage');
title('original signal ');



%% Exercise 2: smoothing and averaging
%1
y3=randn(1,400)/5+ [zeros(1,100) ones(1,200) zeros(1,100)];
%2 snr
noise3=y-[zeros(1,100) ones(1,200) zeros(1,100)];
Snoise3 = std(noise3) %noise powe
Ssignal3 = mean(y3) %signal power

SNRp3 = Ssignal3.^2/Snoise3^2 %SNR power
SNRrms=SNRp3.^0.5 %SNR RMS

%3
avgSig=zeros(1,400);
avgNoise=avgSig;
for ii=1:10
    x=randn(1,400)/5+ [zeros(1,100) ones(1,200) zeros(1,100)];
    avgSig=x.*0.1+avgSig;
    avgNoise=avgNoise+0.1*(x-[zeros(1,100) ones(1,200) zeros(1,100)])
end
SNRpAvg = avgSig.^2/avgNoise.^2 %SNR power
SNRrmsAvg=SNRpAvg.^0.5 %SNR RMS
%4
figure
plot (1:400,y3,'color', 'g')
hold on
plot (1:400,avgSig,'color', 'm')


xlabel('time (sec)');
ylabel(' voltage');
title('original signal and avg signal ');
hold off
%5smoothing with a simple 5 points rectangle window 
%assuming not allowed to use  rectwin() function..
ySmooth=y;
for ii=3:(length(y)-2)
    ySmooth(ii)=0.2*sum(y((ii-2):(ii+2)));
end

figure
plot (1:400,ySmooth,'color', 'b')
hold on
stairs (1:400,ySmooth,'color', 'c')
xlabel('time (sec)');
ylabel(' voltage');
title('smooth signal ');

%% Exercise 3: sampling with noise and stochastic resonance
t=0:0.01:2; y = 10*sin(2*pi*3*t) + 20*sin(2*pi*t*5);
%apllying units:

%amplifing the signal by 1000 times:
amp=1000;
recordingRange=5;
bitResolution=8
SignalRes=recordingRange/amp/(2^bitResolution); %in Volt
%mVolt:
lowestSignal=SignalRes.*(10^6)
m=[max(y) min(y)]
rm=m(1)-m(2)


figure
plot(t,y*1000)

%sampling 
frequency=50;
t2=0:(1/frequency):2;
yS = 10*sin(2*pi*3*t2) + 20*sin(2*pi*t2*5);
figure
subplot(2,2,1)
stem(t2,yS,'m','MarkerEdgeColor','c')
xlabel('time (sec)');
ylabel(' voltage');
title('sample ');

subplot(2,2,2)
stairs(t2,yS)
xlabel('time (sec)');
ylabel(' voltage');
title('sample as sqr wave ');

subplot(2,2,3)
plot(t,y)
xlabel('time (sec)');
ylabel(' voltage');
title('original signal ');

subplot(2,2,4)
stairs(t2,yS,'--')
hold on
plot(t,y,'m')
xlabel('time (sec)');
ylabel(' voltage');
title('original signal and sampled signal ');

%adding noise
yNoise = 10*sin(2*pi*3*t2) + 20*sin(2*pi*t2*5)+20*randn(1,length(t2));
yAvg=(yNoise+yS)/2

figure
subplot(2,2,4)
plot(t2,yNoise,'m')
xlabel('time (sec)');
ylabel(' voltage');
title('signal with noise ');

subplot(2,2,3)
plot(t2,yAvg)
xlabel('time (sec)');
ylabel(' voltage');
title('avg of Signal and noisey signal');

subplot(3,1,1)
plot(t2,yNoise,'--')
hold on
plot(t2,yAvg,'m')
xlabel('time (sec)');
ylabel(' voltage');
title('avg signal and the signal with noise ');


end

