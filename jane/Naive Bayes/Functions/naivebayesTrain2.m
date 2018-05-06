

function model = naivebayesTrain2(data)
%data will be training data 
model ={};
frequency_table={};
featureCount=size(data,2);
dataSize = size(data,1);
class = data(1:dataSize,featureCount); %CLASS IS LAST COL
classes = unique(class);
sizeC = size(classes,1);
feature_probabilities={};
feature_frequencies={};
feature_features={};
empirical_class_prob=zeros(sizeC-1);

for o=1:dataSize
    for c=1:(sizeC-1)
        if (class(o,1)==classes(c,1))
            empirical_class_prob(c,1)=empirical_class_prob(c,1)+1;
        end
    end
end
empirical_class_prob=empirical_class_prob/dataSize;

for n=1:featureCount-1 
feature = data(1:dataSize,n); 
frequency_table(1:4,1) = frequencyTable(feature,class);
feature_probabilities(n) = frequency_table(2,1);
feature_frequencies(n) =frequency_table(3,1);
feature_features(n) = frequency_table(1,1);
end

model(1:5,1)={empirical_class_prob; feature_features; feature_frequencies; feature_probabilities;   classes};


end