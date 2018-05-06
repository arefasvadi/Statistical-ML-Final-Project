function [unimarginals,pairmarginals] = marginals(data)
unimarginals = {};
pairmarginals = {};
features = size(data,2);
observations = size(data,1);
feature_sizes = zeros(features,1);
features_f={};
count_assignment_featurefi = {};
count_assignment_featurefij={};

for f=1:features
    feature_f_vector = data(1:observations,f);
    features_f(1,f) = {unique(feature_f_vector)};
    feature_sizes(1,f) = length(features_f{1,f});
    count_assignment_featurefi(1,f)={zeros(feature_sizes(1,f),1)};
   
end
for f=1:features
    for j=1:features
        count_assignment_featurefij(f,j)={zeros(feature_sizes(1,f),feature_sizes(1,j))};
    end
end

%count of assignments per feature
for o=1:observations
    for f=1:features
        [row,col] = find(features_f{1,f}==data(o,f));
        count_adjusting = count_assignment_featurefi{1,f};
        count_adjusting(row,1)= count_adjusting(row,1)+1;
        count_assignment_featurefi(1,f)={count_adjusting};
        for j=1:features
            [rowi,colx] = find(features_f{1,f}==data(o,f));
            [colj,colz] = find(features_f{1,j}==data(o,j));
            count_adjusting_ij = count_assignment_featurefij{f,j};
            count_adjusting_ij(rowi,colj)= count_adjusting_ij(rowi,colj)+1;
            count_assignment_featurefij(f,j)={count_adjusting_ij};
            
        end
    end
end
pair_marginal_fj = {};
for f=1:features
    unimarginals(f,1) = {count_assignment_featurefi{1,f}/observations};
    unimarginals(f,2) = {features_f{1,f}};
    for j=1:features
        
        pair_marginal_fj(1,1)= {count_assignment_featurefij{f,j}/observations};
        pair_marginal_fj(1,2)= {features_f{1,f}};
        pair_marginal_fj(1,3)= {features_f{1,j}};
        pairmarginals(f,j)={pair_marginal_fj};
    end
end


end
