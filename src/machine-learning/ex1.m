function  ex1()
close all
%Data singnaling ex.1

%A
% (exe1_1Data.mat) to get the variables: t,d, twv, wv and events2
load('exe1_1_data.mat');

%plot(t,d) to see the recording trace (t time [s],d voltage [mV])
figure 
plot(t,d);
xlabel('Time (sec)');
ylabel('Membrane Voltage (mV)');
title('change in voltage over the time in raw data');


% Set threshold for spike detection as 5 times the standard deviation of
% the raw signal

thr=std(d)/1.5;
[~,loc]=findpeaks(d, 'Threshold',thr);
new=zeros(1,500);
figure
y=zeros(1,801);
yA=zeros(1,801);
for ii= 1:length(loc)
    stP=loc(ii)-400;
    endP=loc(ii)+400;
    y=d(stP:endP);
    plot(y)
    hold on
    yA=y+yA;
end
yA=(1/(length(loc))).*yA;
plot(yA,'color' ,'r')
xlabel('Time (msec)');
ylabel('Membrane Voltage (mV)');
title('The spikes waveforms ');
hold off


    





%Plot the isi histograms of the neuron twv 
ISI_datta=diff(twv);
figure 
bar(ISI_datta);
xlabel('Time (mSec)');
ylabel('number of spikes');
title('ISI plot for neuron twv');

%How many spikes do you have in the refractory period
refSpike=length(ISI_datta)-length(diff(twv,2))

%B
%spikes wave forms
figure 
plot(wv','color', 'b')
xlabel('Time (mSec)');
ylabel('Membrane Voltage (mV)');
title('wave form ');
hold on
avgS=mean(wv);
plot(avgS,'color', 'm' );

%raster&psth
rasterMat=zeros(length(events2),1001);

rangeStart=0;
rangeEnd=0;
ind=0;
for ii=1:length(events2);
    rangeStart=events2(ii)-0.5;
    rangeEnd=events2(ii)+0.5;
    for jj=1:length(twv)
        if (twv(jj)<rangeEnd) && (twv(jj)>rangeStart)
           ind=round((twv(jj)-rangeStart)*1000);
           rasterMat(ii,ind)=1;
        end
    end
end
           
figure
subplot(2,1,1);
hold on;
imagesc(-rasterMat)
colormap(gray);
xlabel('Time (ms)');
ylabel('Stim Num');
title('Raster Plot');

    
    
    
avgVec=zeros(1,1001);
x=0;
for ii=1:1001
    x=sum(rasterMat(:,ii));
    x=x/17;
    avgVec(ii)=x;
    
end
    

hold on;
subplot(2,1,2);
plot(avgVec)
xlabel('Time (ms)');
ylabel('mean count per stim');
title('PSTH');

%excitation? Inhibition?
effect='Inhibition';
if sum(avgVec(1:500)) < sum(avgVec(501:end))
    effect='excitation';
end
effect
            
%the mean firing rate 200 ms after the event is on
ans=avgVec(701)

end










