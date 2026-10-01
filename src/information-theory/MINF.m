function [ MI ] = MINF( R )
pX = sum(R,1);
pY = sum(R,2);
MI = 0;
for x = 1:size(R,2)
    for y = 1:size(R,1)
        pXY = R(y,x);
        if pXY && pX(x)
            MI = MI+(pXY*(log2(pXY/(pX(x)*pY(y)))));
        end
    end
end
end

