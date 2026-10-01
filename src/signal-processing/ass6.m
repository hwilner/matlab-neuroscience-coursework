close all
load('exercise6_data.mat');

Yhist = hist(train_Y,8)/length(train_Y);

%% I

SR = zeros(22,22);
R = cell(22,22);
for ii = 1:22
    R1 = joint(train_X{ii}, train_Y);
    MIR1 = MINF(R1);
    for jj = 1:22
        R2 = joint(train_X{jj}, train_Y);
        MIR2 = MINF(R2);
        R1and2 = joint(train_X{ii}, train_X{jj});
        R{ii,jj} = R1and2;
        % I(R1,R2;S) = H(R1,R2)-H(R1,R2|S)
        ind = (find(R1and2~=0))';
        HR1and2 = -sum(sum(R1and2(ind).*(log2(R1and2(ind)))));
        HR1and2givenS = zeros(1,8);
        for kk = 1:8
            new_R1and2 = joint(train_X{ii}(find(train_Y==kk)), train_X{jj}(find(train_Y==kk)));
            ind = (find(new_R1and2~=0))';
            HR1and2givenS(kk) = -sum(sum(new_R1and2(ind).*(log2(new_R1and2(ind)))));
        end
        HR1and2givenS = sum(HR1and2givenS.*Yhist);
        MIR1and2 = HR1and2 - HR1and2givenS;
        SR(ii,jj) = MIR1and2 - (MIR1 + MIR2);
    end
end

figure;
imagesc(SR);
colorbar('vert');

SRtri = triu(SR,1);
redundancy = length(find(SRtri<0));
synergy = length(find(SRtri>0));
SRsum = sum(sum(SRtri));

%% II

noiseCorr = zeros(22,22);
for ii = 1:max(train_Y)
    for jj = 1:22
        for kk = 1:22
            j = joint(train_X{jj}(find(train_Y==ii)),train_X{kk}(find(train_Y==ii)));
            noiseCorr(jj,kk,ii) = (MINF(j)).*Yhist(ii);
        end
    end
end

noiseCorr = sum(noiseCorr,3);
noiseCorr2 = noiseCorr - diag(diag(noiseCorr));
[x,y] = find(noiseCorr2==max(max(noiseCorr2)));
regress = zeros(1,8);
figure
for ii = 1:8
    jj = ii;
    if ii>4
        jj = jj+1;
    end
    if jj~=5
        subplot(3,3,jj);
        scatter(train_X{x(1)}(find(train_Y==ii)),train_X{y(1)}(find(train_Y==ii)),30,[0.5 0 0.5],'d','fill');
        istr = num2str(ii);
        title(['direction ', istr]);
        axis('square');
        [reg,~,~] = regression(train_X{x(1)}(find(train_Y==ii)),train_X{y(1)}(find(train_Y==ii)));
        regress(ii) = reg;
    end
    subplot(3,3,5);
    %% Julia set
    col=30;
    m=400;
    cx=0;
    cy=0;
    l=1.5;
    xx=linspace(cx-l,cx+l,m);
    yy=linspace(cy-l,cy+l,m);
    [X,Y]=meshgrid(xx,yy);
    c= -0.561321+0.641000*i;
    Z=X+i*Y;
    for k=1:col;
        Z=Z.^2+c;
        W=exp(-abs(Z));
    end
    %colormap prism(256)
    pcolor(W);
    shading flat;
    axis('square','off');
end

for ii = 1:22
    for jj = 1:22
        signalCorr(ii,jj) = MINF(R{ii,jj});
    end
end

SR2 = noiseCorr - signalCorr;
figure;
imagesc(SR2);
colorbar('vert');
