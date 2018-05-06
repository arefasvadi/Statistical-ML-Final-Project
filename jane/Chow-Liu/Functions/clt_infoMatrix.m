function infoMatrix= clt_infoMatrix(data)
%need to include class as a feature in this method
featureCount = size(data,2);
observations = size(data,1);
infoMatrix = zeros(featureCount,featureCount);
%for each pair of features (not equal) fill in the i,j entry with the
%mutual information
for i=1:featureCount
    for j=i+1:featureCount
        %mutual info on ith and jth columns of the data
        infoMatrix(i,j) = mutualInfo(data(1:observations,i),data(1:observations,j));
    end
end

end