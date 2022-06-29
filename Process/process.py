import os
from Process.dataset import BiGraphDataset
cwd=os.getcwd()


def loadTree(dataname):

    treePath = os.path.join(cwd,'data/'+dataname+'/data.TD_RvNN.vol_5000.txt')
    print("reading twitter tree")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}

    return treeDic


def loadBiData_graph(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    data_path = os.path.join(cwd,'data', dataname + '_graph')
    print("loading train set", )
    traindata_list = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

