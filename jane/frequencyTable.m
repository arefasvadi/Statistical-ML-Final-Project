function frequencytable = frequencyTable(feature, classification)
% input feature vector, and corresponding classification
%

frequencytable = {};
features = unique(feature);
size_i = size(unique(feature),1);
observations = size(feature,1);
size_C = size(unique(classification),1);
classes=unique(classification);
classifications=classes(1,1);
probabilities = zeros(size_i,size_C);
frequencies = zeros(size_i,size_C); 
extra = size_i*(size_C);
count_observations=zeros(size_C,1);
extra_byClass = zeros(size_C,1);
for o=1:observations
    %calculate frequency
    for c=1:size_C
        if (classification(o,1)==classes(c,1))
            count_observations(c,1)=count_observations(c,1)+1;
            extra_byClass(c,1)= size_i;
        end
    end
    for i=1:size_i
        
        for c=1:(size_C)
            %at each observation look for each combinatin of
            %feature-classification and it to the appropriate place in the
            %frequency table.
            if and(feature(o,1)==features(i,1),(classification(o,1)==classes(c,1)))
                 %mushroom data will just have p(i)  and p(c) 
                    %vector of i
                    %offset of i*(c-1)
                    %frequencies(i+i*(c-1),1)
                    %frequencies(i,1) =frequencies(i+i*(c-1),1)+1;
                   
                frequencies(i,c) = frequencies(i,c)+1;
            end
        end
        
    end
end%loop through observations

%frequencies is fully resolved
%probabilities:
for p=1:size_i
    for c=1:size_C
    probabilities(p,c)=(frequencies(p,c)+1)/(extra_byClass(c,1)+count_observations(c,1));
    end
end
%summing to 1. should sum to C. need p(feature|classification).


%output: (p1...pn), (x1....xn), (k1,...,kn) n categories of feature x.
%corresponding to probability pi feature xi occurs for classification k, K
%classifications.
%incorporate laplace smoothing here
%add 1 to the count of observations of xi.
%divide by |v| extra in observations v=observations
frequencytable(1:4,1) = {features; probabilities; frequencies; classes};
end