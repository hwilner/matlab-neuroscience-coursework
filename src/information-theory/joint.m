function [ R ] = joint( X, Y )
R = zeros(max(Y),max(X)+1);

for j = 1:(max(Y)+1)
    j = j-1;
    minimum = min(X(find(Y==j)))+1;
    maximum = max(X(find(Y==j)))+1;
    n_bins = maximum-minimum+1;
    R(j+1,minimum:maximum) = hist(X(find(Y==j)), n_bins)/(length(Y));
end

if size(Y,2)==1
    R(1,:) = [];
end
end