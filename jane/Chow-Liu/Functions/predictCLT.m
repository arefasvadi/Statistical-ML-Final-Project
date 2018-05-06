% predict y-label using the model; 
function [classifications, accuracy] = predictCLT(unimarg, pairmarg, tree, feature, test)
    %pull in test data
    %test data will not have the column of classification feature.
    %test data will have other feature data
    
    %get neighbors of feature node specified for classification
    %neibrs = neighbors(tree,1);
    preds = predecessors(tree,feature);
    
    %check if either are empty 
    if (~isempty(preds))
        neibrs = preds';
    else
        return;
    end
            
    observations = size(test,1);
    actual = test(1:observations,feature);
    error = observations;
    
    degree = numel(neibrs);
    unim_features = unimarg{feature,2}; %classes vector
    score = zeros(1,numel(unim_features)); %size of classes
    unim_dist = unimarg{feature,1}; %marginals for classes
    classifications = zeros(observations,1);
    for o=1:observations
    for i = 1:length(score) % cycle through all possible classifications
        score(i) = (1 - degree)*log(unim_dist(i));
        for j=1:degree % cycle through all the neighbors of class node
            f_j = neibrs(j); %col of interest from predecessors
            feature_observed = test(o,f_j); %observed value in data
            pm = pairmarg{feature,f_j}; %marginal distribution for pair
            pm_dist = pm{1,1};
            pm_i=pm{1,2};
            pm_j=pm{1,3};
            
            pb = pm_dist(pm_i==unim_features(i),pm_j==(feature_observed)); %marginal pair dist
            if (isempty(pb)) 
                pb = numel(pm_dist)/(numel(pm_dist)+observations); % if it didn't show up in training do a thing
            end
            if (pb~=0)
                score(i) = score(i) + log(pb);
            end
               
        end
    end
     [~,idx] = max(score);
    
    classifications(o,1) = unim_features(idx);
    if (classifications(o,1) == actual(o,1))
        error = error-1;
    end
   
    end
    
     accuracy = (1 - error/observations)*100;
end
        
