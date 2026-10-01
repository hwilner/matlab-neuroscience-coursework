function [cells_mean, S1, S2] = COMP3(Cell1, Cell2)

cells = [Cell1 ; Cell2];

%% I
cells_mean = mean(cells,2);

%% II
n_bins = max(max(cells))+1;
cells_dist = ((hist(cells', n_bins))')/size(cells,2);

figure;
subplot(2,2,1);
plot([0:n_bins-1], cells_dist(1,:), 'color', [0.3,0.75,0.94]);
hold on
plot([0:n_bins-1], cells_dist(2,:), 'color', [0.49,0.18,0.56]);
title('cell 1 spikes distribution', 'FontSize', 12);
xlabel('spike rate');
ylabel('%');
legend('condition1', 'condition2');

subplot(2,2,2);
plot([0:n_bins-1], cells_dist(3,:), 'color', [0.3,0.75,0.94]);
hold on
plot([0:n_bins-1], cells_dist(4,:), 'color', [0.49,0.18,0.56]);
title('cell 2 spikes distribution', 'FontSize', 12);
xlabel('spike rate');
ylabel('%');
legend('condition1', 'condition2');

%% III
thresh = zeros(size(cells,1),n_bins);
for jj = 1:4
    for ii = 1:(n_bins)
    thresh(jj,ii) = sum(cells_dist(jj,ii:end));
    end
end

subplot(2,2,3);
plot(thresh(2,:), thresh(1,:), 'color', [0.87,0.49,0]);
title('cell 1 ROC curve', 'FontSize', 12);
xlabel('False Positive');
ylabel('True Positive');
axis([0,1,0,1]);

subplot(2,2,4);
plot(thresh(4,:), thresh(3,:), 'color', [0.87,0.49,0]);
title('cell 2 ROC curve', 'FontSize', 12);
xlabel('False Positive');
ylabel('True Positive');
axis([0,1,0,1]);

%% IV
S1 = abs(trapz(thresh(2,:), thresh(1,:)));
S2 = abs(trapz(thresh(4,:), thresh(3,:)));

end