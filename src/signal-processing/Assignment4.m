function Assignment4
close all
%% 1
tX=0:0.001:1;
x= sin(2*pi*3*tX);
tY=0:0.001:2;
y= sin(2*pi*3*tY+pi/3);

% a

myFunc=crossCorrelation(x,y);
compersaion=compareCC(myFunc,x,y);

%b
plotCC(x);
plotCC(y);

%% 2
load('spikeTrain.mat')
figure
for i=1:size(spikeTrain,1)
    j=i+1;
    if j>size(spikeTrain,1)
        j=1;
    end
    y = fliplr(xcorr(spikeTrain(i,:), spikeTrain(j,:), 500));
    subplot(3,2,2*i-1)
    colorp=[0 0 0]
    colorp(i)=1;
    colorp(j)=1;
    stairs(y,'Color' ,colorp*0.1)
   
     y = fliplr(xcorr(spikeTrain(j,:), spikeTrain(i,:), 500));
    subplot(3,2,2*i)
    colorp=[0.5 0.5 0.5]
    colorp(i)=1;
    colorp(j)=0;
    stairs(y,'Color' ,colorp*0.5)
   
end
%% 3
Eggs=[5777 4225 2674 1249 749 870];
dom=1:6;
[Pearson,Spearman ssxx,ssyy,ssyx,rValue ]=monkeyParty(Eggs,dom)
end


%% q1 functions


function [cc] = crossCorrelation(A,B)
N = size(A,2);
M = size(B,2);
result = zeros(1, N + M - 1 );
len = size(result,2);

for m = 1 : len
    arg = (m - N);
    
    if(arg < 0)
        negativeCondition = 1;
        limit = N + arg;
    else
        negativeCondition = 0;
        limit = N - arg;
    end
    
    for n = 1:limit
        if(negativeCondition == 0)
            result(m) = result(m) + A(arg + n) + B(n);
        else
            result(m) = result(m) + A(n) + B(n - arg);
        end
    end
end
cc=result


end

function [comp] = compareCC(myFunc,xV,yV)
matFunc=fliplr(xcorr(xV,yV));
if length(myFunc)==length(matFunc)
    c=sum(matFunc-myFunc)/length(myFunc)*100;
    if c>0
        comp= 100- c;
    else
        comp=100+c;
    end
else
    comp=0;
end

figure
plot(matFunc,'r')
figure
plot(myFunc)
end



function plotCC(vec)
cc=crossCorrelation(vec,vec)
figure
plot(cc)
hold on
plot(fliplr(xcorr(vec)),'m')
end

%% q2 functions
%not eficient run over 20 min!!!
function [wcc] = window( x )
[m,n]=size(x);
wcc=zeros(m*m,length([zeros(1,500) x(1,:) zeros(1,500)]));
figure
for i=1:m
    j=i+1;
    if j>m
        j=1;
    end
    ni=[zeros(1,500) x(j,:) zeros(1,500)];
    nj=[zeros(1,500) x(i,:) zeros(1,500)];
    new=[zeros(1,500) x(i,:) zeros(1,500)];
    new2=new;
    for ii=500:length(new)
        ij=ii-499;
        ji=ii+499;
        if ni(ii)==1
            new(ij:ji)=new(ij:ji)+nj(ij:ji);
            if nj(ii)==1
                new2(ij:ji)=new(ij:ji)+ni(ij:ji);
            end
            wcc(2*i-1,:)=new(:);
            wcc(2*i,:)=new2(:);
            
        end
     subplot(3,2,2*i)
    colorp=[0.4 0.4 0.6]
    colorp(i)=1;
    colorp(j)=0;
    plot(new2,'Color' ,colorp)
    subplot(3,2,2*i-1)
    colorp=[0.8 0.2 0.1]
    colorp(i)=1;
    colorp(j)=1;
    plot(new,'Color' ,colorp)
    
    end
end
    
    
    
end


%% q3 functions:

function [pearson,spearman ssxx,ssyy,ssxy,rValue ]=monkeyParty(a,b)
[RHO,PVAL] = corr(a',b','Type','Spearman')
spearman=[RHO,PVAL];
[RHO,PVAL] = corr(a',b','Type','Pearson')
pearson=[RHO,PVAL];
aNor=a-mean(a);
bNor=b-mean(b);
ssxx=(sum(aNor.^2))^0.5;
ssyy=(sum(bNor.^2))^0.5;
ssxy=sum(aNor.*bNor)
rValue=ssxy/(ssxx*ssyy)

end