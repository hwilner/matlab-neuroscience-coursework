
%% 1
%a table
diet=[ 10 2 3 4 5 1;
    6 3 1 6 7 2;
    8 1 1 3 5 4;
    0 2 2 6 6 3;
    5 2 1 6 6 3;
    10 6 7 3 4 8;
    3 6 7 1 4 5;
    3 7 5 1 2 5;
    8 4 6 3 2 6;
    0 9 7 1 4 4];
UN={'England'; 'Finland'; 'Sweden'; 'Denmark'; 'Germany' ;'Portugal' ;'Spain' ;'Italy'; ...
    'Greece' ;'Isreal'};
%plot origin data
figure
subplot(221)
plot(1:6,diet(:,:),'-d')

legend(UN{:},'Location','westoutside')
title('a:original data')
set(gca,'Xtick', 1:6,'XTickLabel',{'coffee' 'garlic' 'spices' 'meat' 'fish' 'chocolate'})
axis('square');

%pca

dietN= bsxfun(@minus,diet,mean(diet,2)); %subtract each row  by its mean
dietC=cov(dietN);
[eigVec, eigValMat] = eig(dietC);



eigVal = diag(eigValMat);
value1=find(eigVal==max(eigVal));
eigVal(value1)= 0;
value2= eigVal==max(eigVal);
PC1=eigVec(:,value1);
PC2=eigVec(:,value2);

PCx=diet*PC1;
PCy=diet*PC2;
%c plot pca

subplot(222)
scatter(PCx,PCy);
axis('square');
title('b:the data on 2 components plane');
xlabel('PC1');ylabel('PC2');

%(1,3),(6,9), (8,7),(2,5)
subplot(224)
hold on
plot(PC1,'m');plot(PC1,'dg');
plot(PC2,'c');plot(PC2,'ob');
title(' d: PCA1 and PCA2 ');
axis('square');
set(gca,'Xtick', 1:6,'XTickLabel',{'coffee' 'garlic' 'spices' 'meat' 'fish' 'chocolate'})
xlabel('coeffients of original dimensions');ylabel('original dimensions');
legend('pc1','pc2','Location','westoutside')

%b percentage and accumulating variance graph
eigVal = sort(diag(eigValMat),'descend');
cumulativePcVar=100*(cumsum(eigVal)/sum(eigVal));
%cumplotive sum div in sum%to re erange thepc from highest to  lowest.
subplot(223)
area(cumulativePcVar)
axis('square');
title('c:percentage and accumulating variance graph');
xlabel('property');ylabel('accumulating variance');


%% 2
load('d2.mat');
%a ploting
figure
subplot(221)
plot(t,d2);
title('a: raw data ');
xlabel('time [S]');ylabel('voltage [mV]');
%b :Filtering the noise (frequency below 300)

Fs=30000;
NFFT = length(d2);
F = ((0:1/NFFT:1-1/NFFT)*Fs).';
D2=fft(d2,NFFT);
D2(F<=300 | F>=Fs-300) = 0;
d2=ifft(D2,'symmetric');
subplot(222)
plot(t,d2);
title('b: signal after applying high  pass fillter ');
xlabel('time [S]');ylabel('voltage [mV]');
%c spike  extraction
%thr=-std(d2)/(mean(d2)*170);
%[~,loc]=findpeaks(d2, 'Threshold',thr);
%treshMat=zeros(length(loc),32);
%for ii=1:length(loc)
%  treshMat(ii,:)=d2(loc(ii)-16:loc(ii)+15);
%end


d = d2-(-0.1);
S = [0 d(1:end-1).*d(2:end)];
indx = find(S < 0 & d < 0 );
wv=[];
for i=1:length(indx)
    wv(end+1,:) = d(indx(i)-16:indx(i)+15);
end
subplot(223)
plot(0:1/32:1-1/32,wv')
title('c: wave form ');
xlabel('time [mS]');ylabel('voltage [mV]');
%d
treshMat=wv';
norm=mean(treshMat);

treshMat= bsxfun(@minus,treshMat,mean(treshMat)); %subtract each row  by its mean
treshMat=cov(treshMat);
[eigVec, eigValMat] = eig(treshMat);
eigVal = diag(eigValMat);
value1=find(eigVal==max(eigVal));
eigVal(value1)= 0;
value2=find(eigVal==max(eigVal));
PC1=eigVec(:,value1);
PC2=eigVec(:,value2);
PCx=treshMat*PC1;
PCy=treshMat*PC2;


%% choose of data 
treshMat=[PCx, PCy]';
data_dim = size(treshMat,1);
%% definig number of clusters
k=2;% row of centers



%% choosing random centers 
centers = zeros(k, data_dim);
for i=1:k
    for j=1:data_dim
        centers(i,j)=((max(treshMat(j,:))-min(treshMat(j,:))).*rand(1,1) + min(treshMat(j,:)));
    end
end

%centers=[0.05 0.05; 1.4 0]
%% intalizing variables
run=0;change=0;
%Same=0;
X=ones(k,size(treshMat,2));
Y=ones(k,size(treshMat,2));
l=length(X(1,:));
kV=1:k;
assign=randi(kV,1,l);


oklide=zeros(k,length(X(1,:)));
minimal=ones(1,length(X(1,:)));
oklide2=oklide';
newC=[0 0; 0 0];

%% 

while ~isequal(centers, newC)
    %%  distance from centers
    for i=1:k
       X(i,:)=treshMat(1,:)-centers(i,1);
        Y(i,:)=treshMat(2,:)-centers(i,2);
        for j=1:length(X(1,:))
            oklide(i,j)=(X(i,j)^2+Y(i,j)^2)^0.5;
        end
    end
    
    %% assign to closest center
    for i=1:l
        if oklide(1,i)<oklide(2,i)
            assign(i)=1;
        else
            assign(i)=2;
        end
    end
   
   %% new centers
   for j=1:data_dim
       vec=treshMat(j,:);
       for i=1:k
        newC(i,j)=mean(vec(find(assign==i)));
       end
   end
   
   %% cheking if centers rep. itself
   %Same=min(min(centers==newC));
    centers=newC;
   
   

end

%% ploting 
neXk1=PCx(find(assign==1));   neYk1=PCy(find(assign==1));
neXk2=PCx(find(assign==2));   neYk2=PCy(find(assign==2));
 

subplot(224)
hold on
title(['d:spikes PCA, and K-mean  ']);
xlabel('PC1');ylabel('PC2');
scatter(neXk1,neYk1,30,[0.5 0 0.5],'d','fill');
scatter(neXk2,neYk2,30,[1 0 1],'c','fill');
axis([-0.3 1.3 -0.1 0.2])

%% linear superation line by SVM
% Orthogonal toline between the closet points from the difreent clusters 
lineM=((neXk1(1)-neXk2(1))^2+(neYk1(1)-neYk2(1))^2)^0.5;
for ii=1:k
    for  jj=1:length(neXk1)
        for kk=1:length(neXk2)
            line= ((neXk1(jj)-neXk2(kk))^2+(neYk1(jj)-neYk2(kk))^2)^0.5;
            if line<lineM
                centers=[kk jj];
            end
        end
    end
end
% ploting sup. vec
xCor=[neXk1(centers(2)),neXk2(centers(1))];
yCor=[neYk1(centers(2)),neYk2(centers(1))];
plot(xCor,yCor,'--m');
midPoint=([sum(xCor),sum(yCor)])/2;
slope=(neYk1(centers(2))-neYk2(centers(1)))/(neXk1(centers(2))-neXk2(centers(1)));

%ploting Perpendicular Bisector line
PBslope=-1/slope;
lengthPBLine=2*max(max(oklide));
xLine = (midPoint(1)-lengthPBLine:midPoint(1)+lengthPBLine);
yLine =( PBslope*(xLine-midPoint(1))+ midPoint(2));

plot(xLine,yLine,'b')
