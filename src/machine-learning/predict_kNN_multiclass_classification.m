function [classification_prediction, performance_prediction] = predict_kNN_multiclass_classification(k, test_X, train_X, train_Y)
    % Classify a given test set based on k-nearest-neighbors algorithm.
    % Return classification and estimation regarding its performance. 

    %% Initialize variables
    n = size(train_Y,2);
    number_of_folds = 7;
    k = 1;
    
    %% kNN cross-validation
    validation_performance = zeros(1,number_of_folds);
    for i = 1:number_of_folds  
        seg_size = n/number_of_folds;
        training_X = train_X;
        training_X(:,((i-1)*seg_size+1):i*seg_size) = [];
        training_Y = train_Y;
        training_Y((i-1)*seg_size+1:i*seg_size) = [];
        validation_X = train_X(:,((i-1)*seg_size+1):i*seg_size);
        validation_Y = train_Y((i-1)*seg_size+1:i*seg_size);
    
        classification = get_kNN_classification (k, validation_X, training_X, training_Y);
        validation_performance(i) = (length(find(classification==validation_Y)))/length(classification);
    end

    %% Return values
    classification_prediction = get_kNN_classification(k, test_X, train_X, train_Y);
    performance_prediction = mean(validation_performance);
end
