function [CLTree, infoMatrix, unim, pairm] = ChowLiuTree(data)
%data is a categorical or numerical nxm array of data with features as
%columns


infoMatrix = clt_infoMatrix(data);
clt_info_wts = -1*infoMatrix;
%get tree from spanning tree bult in.
G = graph(clt_info_wts,'upper');
[T, pred] = minspantree(G,'Type','forest','Root',findnode(G,1));

rootedTree = digraph(pred(pred~=0),find(pred~=0),[]);


CLTree = rootedTree;
[unim, pairm] = marginals(data);


end