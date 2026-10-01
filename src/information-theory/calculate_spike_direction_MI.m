function [MI, the_chosen_one] = calculate_spike_direction_MI (train_X, train_Y)

    % Initialize variables
    numOfNeurons = size(train_X,1);
    MI = zeros(1,numOfNeurons);
    R = cell(numOfNeurons, 1); 
    
    % Per each neuron
    for i = 1:numOfNeurons
        R{i} = zeros(8, max(train_X{i})+1);
       for j = 1:8  
            % Create joint probabity tabler
            minimum = min(train_X{i}(find(train_Y==j)))+1;
            maximum = max(train_X{i}(find(train_Y==j)))+1;
            n_bins = maximum-minimum+1;
            R{i}(j,minimum:maximum) = hist(train_X{i}(find(train_Y==j)), n_bins)/...
                (length(train_Y));
       end
       % Calculate mutual information
       pX = sum(R{i,1},1);
       pY = sum(R{i,1},2);
       for x = 1:size(R{i,1},2)
           for y = 1:8
               pXY = R{i,1}(y,x);
               if pXY && pX(x)
                   MI(i) = MI(i)+(pXY*(log2(pXY/(pX(x)*pY(y)))));
               end
           end
       end
    end
    the_chosen_one = find(MI==max(MI));
end