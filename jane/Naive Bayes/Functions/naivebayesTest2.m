
function classifications = naivebayesTest2(model, data)
%data will be testing data

%adjust test data for new feature observations in multinomial data???

%laplace smoothing... will need to resmooth previous frequency tables.


featureCount=size(data,2);
dataSize = size(data,1);
class_index = featureCount; %held in last column of dataset
class_ACTUAL = data(1:dataSize,featureCount); 
class_PREDICTED = zeros(dataSize,1);
classes = model{5,1};
classSize = size(classes,1);
alpha = ones(featureCount,classSize);
%get data from model
%really a lot of checks SHOULD happen, but we'll trust the user this
%time...
empirical_prob = model{1,1};

pof_class_given_obs = zeros(classSize,1);
pof_feature_f_given_c =zeros(classSize,1);


features_cell = model{2,1};
probabilities = model{4,1};
frequencies = model{3,1};


%how many extra features showed up in test data than were in training?

for f=1:featureCount-1
    features_training = features_cell{1,f};
    features_testing = unique(data(1:dataSize,f));
    all_the_features = unique([features_training;features_testing]);
    features_probabilities = probabilities{1,f};
    values = size(features_probabilities,1);
    %observations in class c
    frequencies_training = frequencies{1,f};
    f_class = sum(frequencies_training);
    for c=1:classSize %divide 1 by class observations and new additional observations in test data
        alpha(f,c) = 1/(f_class(1,c)+(length(all_the_features)-length(features_training)));
        for v=1:values
            %adjust previous probability table calculations by alpha
            features_probabilities(v,c)=features_probabilities(v,c)-alpha(f,c);
        end
    end
end

error = dataSize;
%multinomial feature distributions: p(classification| evidence) =
%
% p(Classification)* product over all features: [ p(feature| classification)]
for o=1:dataSize
    %for each test case get classification prediction

    for f=1:featureCount-1
         %for each feature, get probability of that feature for the
            %classification, 'e'.
          %which features index?

          [prow,col] = find(features_training == data(o,f)); %row gives value of significance in prob matrix
          %[row,pcol] = find(classes == data(o,class_index)); %col gives value of significance in prob matrix
          PI_pof_feature_given_c = zeros(classSize,1);
          %create a vector of size c and compare end results
          for c=1:classSize
              if (isempty(prow))% haven't seen this feature's value before... non-zero prob?
                  %laplace smoothing alpha (sums to 1 by feature and class)
                  
                  pof_feature_f_given_c(1,1)= alpha(f,c);
                  
              else
                pof_feature_f_given_c(1,1) = features_probabilities(prow,c);
              end
           if ( PI_pof_feature_given_c(c,1)==0)
                PI_pof_feature_given_c(c,1)= pof_feature_f_given_c(1,1);
           else
                PI_pof_feature_given_c(c,1) = PI_pof_feature_given_c(c,1) * pof_feature_f_given_c(1,1);
           end
          end         
    end
    %which is larger probability of 2 classes?
    %PI holds 2 columns representing feature probabilities for the 2
    %classes (or 1 column per class if more than 2).
    class_chosen=1;      
    maxprob = 0;
    for c=1:classSize
        if (PI_pof_feature_given_c(c,1)>maxprob)
            maxprob = PI_pof_feature_given_c(c,1);
            class_chosen = c;
            class_PREDICTED(o,1) = class_chosen;
        end
    end
    pof_class_given_obs(o,1)=empirical_prob* PI_pof_feature_given_c(class_chosen,1);
   
    
    if (class_ACTUAL(o,1)==class_PREDICTED(o,1))
        error = error-1;
    end
    
end

accuracy = (1-error/dataSize)*100;



classifications(1:4,1)={class_ACTUAL,class_PREDICTED, pof_class_given_obs, accuracy};

end