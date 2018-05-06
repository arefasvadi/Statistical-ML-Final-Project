Workflows:
Naive Bayes
model = naivebayesTrain2(train_data)
model is a 1 by 5 cell representing:
model(1:5,1)={empirical_class_prob; feature_features; feature_frequencies; feature_probabilities;   classes};

classifications = naivebayesTest2(model, test_data)
classifications is a 1 by 4 cell
classifications(1:4,1)={class_ACTUAL,class_PREDICTED, pof_class_given_obs, accuracy};


chow liu tree
[CLTree, infoMatrix, unim, pairm] = ChowLiuTree(train_data)

CLTree is the unadjusted but directed edge tree

the info matrix is the mutual information matrix (positive)

unim is a |features| by 2 matrix of cells containing unimarginal info.
the first column has the  |feature_i| length vector of probability distributions.
the second column has the list of feature i values.

pairm is a |features| by |features| matrix of cells containing pair-marginal info.
the pair-marginal information at each join of features is:
the pair-distribution, the i-feature-set, the j-feature-set.
the pair-distribution is a |feature_i| by |feature_j| matrix of joint probabilities
the i-feature-set is the list of unique values taken by feature i
similarly for j-feature-set.


[classifications, accuracy] = predictCLT(unimarg, pairmarg, tree, classification_feature_index, test_data)
**the tree input to this function needs to have the edges directed to the classification_feature_index value input by the user.