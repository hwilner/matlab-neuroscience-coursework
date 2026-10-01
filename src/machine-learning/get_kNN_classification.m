function test_Y = get_kNN_classification (k, test_X, training_X, training_Y)
    % Classify a given test set based on k-nearest-neighbors algorithm
    % on a given training set. 
    
    %% Initialize variables
    n = size(training_Y,2);
    test_Y = zeros(1,size(test_X,2));

    %% Classify samples one at a time
    for i = 1:length(test_X)

        % Compute all distances from current sample
        distances = zeros(1,n);
        for t = 1:n
            distances(t) = compute_distance(test_X{i}, training_X{t});
        end

        % Assign a label according to the k nearest neighbors
        training_Y_copy = training_Y;
        distances_copy = distances;
        nearest_neighbors = zeros(1,k);
        for j = 1:k
            closest = find(distances_copy==min(distances_copy));
            nearest_neighbors(j) = training_Y_copy(closest(1));
            distances_copy(closest(1)) = [];
            training_Y_copy(closest(1)) = [];
        end
        
        neighbors = zeros(1,7);
        for j = 1 : 7
            neighbors(j) = length(find(nearest_neighbors==j));
        end
        
        best_neighbor = find(neighbors==max(neighbors));
        if length(best_neighbor)>1
            best_neighbor = training_Y(find(distances==min(distances)));
        end
        
        test_Y(i) = best_neighbor;
    end
end

%% Compute the distance between two data samples
function distance = compute_distance(sample1, sample2)
    distance = sqrt(sum((sum(sample1,2)-sum(sample2,2)).^2));
end    
