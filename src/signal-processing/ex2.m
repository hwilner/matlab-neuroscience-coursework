function  ex2(frequency ,sample )
%function ex2 plot sin signal
%the function rescive two input arguments:signal frequency in Hz, and
% and number of points to sample in sec
if frequency<0
    frequency=frequency*-1;
end

x = linspace(0,1,80000);
y = sin(2*pi*frequency*x);



figure
subplot(2,2,1)
plot (x,y,'color', 'm')
xlabel('time (sec)');
ylabel(' voltage');
title('sin signal ');
timeD=1/sample;


subplot(2,2,2)
plot (x,y, 'color', 'm')
hold on
x2 = linspace((0+1/sample),(1-1/sample),sample);
y2=sin(2*pi*frequency*x2);
 stem(x2,sin(x2))
xlabel('time (sec)');
ylabel(' voltage');
title('sampeling  ');


subplot(2,2,3)
plot (x,y,'color', 'm')
hold on
x2 = (linspace((0+1/sample),(1-1/sample),sample)).*rand(1,5);
y2=sin(2*pi*frequency*x2);
stem(x2,y2)
xlabel('time (sec)');
ylabel(' voltage');
title('random sampeling  ');

subplot(2,2,4)
stem(x2,y2)
xlabel('time (sec)');
ylabel(' voltage');
title('The sampeling alone ');
end

