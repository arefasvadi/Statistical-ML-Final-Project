function info = mutualInfo(vector_i, vector_j)
%given a vector i and vector j of data, what is their mutual information?
%how many times are they observed at the same time?
%how many times are they observed individually?

count_unique_features_i = size(unique(vector_i),1);
count_unique_features_j = size(unique(vector_j),1);

unique_features_i = unique(vector_i);
unique_features_j = unique(vector_j);
observations = size(vector_i,1);

Counts_Assignments = zeros(count_unique_features_i,count_unique_features_j);
Count_i = zeros(count_unique_features_i,1);
Count_j = zeros(count_unique_features_j,1);


for o=1:observations
    [row_i,col_i] = find(unique_features_i == vector_i(o,1)); %which feature of i?
    Count_i(row_i,1) = Count_i(row_i,1)+1;
    [row_j,col_j] = find(unique_features_j == vector_j(o,1)); % which feature of j?
    Count_j(row_j,1) = Count_j(row_j,1)+1;
    Counts_Assignments(row_i,row_j)=Counts_Assignments(row_i,row_j)+1;
end

MutualInfo = 0;

for i=1:count_unique_features_i
     for j=1:count_unique_features_j
        N = Counts_Assignments(i,j)/observations;
        N_I = Count_i(i,1)/observations;
        N_J = Count_j(j,1)/observations;
        if (N>0)
            MutualInfo=MutualInfo+N*log10(N/(N_I*N_J));
        end
     end
end


info= MutualInfo;

end