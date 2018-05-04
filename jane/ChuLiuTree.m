function CLTree = ChuLiuTree(data)
%data is a categorical or numerical nxm array of data with features as
%columns
infoMatrix = clt_infoMatrix(data);
%get tree from spanning tree bult in.
[und_tree, cost] = UndirectedMaximumSpanningTree(infoMatrix);
%don't need cost, really
% add directions to tree:
CLTree = und_tree;

end