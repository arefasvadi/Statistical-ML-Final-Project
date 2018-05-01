

function model = naivebayesTrain2(data)
%data will be training data - matrix input of features, and classification
%output will be model to use
model ={};
frequency_table={};
%it will consist of 2 data structures: 
%   a table of probabilities for the features dependent on classification
%   the probabilities of the classification
%mushroom data: 
%   a table of probabilities pi (just 1 per feature per value exhibited by
%   feature.

featureCount=size(data,2);
dataSize = size(data,1);
class = data(1:dataSize,featureCount); %CLASS IS LAST COL
classes = unique(class);
sizeC = size(classes,1);
feature_probabilities={};
feature_frequencies={};
feature_features={};
%   the probability of mushroom = edible empirically

empirical_class_prob=zeros(sizeC-1); %never assume...
%get empirical probability of mushroom edibility:
for o=1:dataSize
    for c=1:(sizeC-1)
        if (class(o,1)==classes(c,1))
            empirical_class_prob(c,1)=empirical_class_prob(c,1)+1;
        end
    end
end
empirical_class_prob=empirical_class_prob/dataSize;

for n=1:featureCount-1 % cycle n through feature data.start with 4 instead of dataSize
feature = data(1:dataSize,n); 
% classification is 1 for mushroom data.
%calculate frequencies table for each feature vector call with (feature,
%classifcation subset of data matrix
frequency_table(1:4,1) = frequencyTable(feature,class);
%[feature_features, feature_frequencies, feature_probabiliities, classes]=frequencyTable(feature,class);
feature_probabilities(n) = frequency_table(2,1);
feature_frequencies(n) =frequency_table(3,1);
feature_features(n) = frequency_table(1,1);
end
%once we have the frequencies per feature-class pairing, construct pi
%table. and calculate the empirical probability the mushrooms are edible:



%model is a frequency table of pi-k  and p(Ck)
%mushroom data classifier is bernoulli so just p(C) and p(xi|c) 

model(1:5,1)={empirical_class_prob; feature_features; feature_frequencies; feature_probabilities;   classes};


end