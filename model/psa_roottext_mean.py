import argparse
import math
import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.load_data import *
import pickle
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='Twitter16', type=str)
parser.add_argument('--model_name', default='PSA', type=str)
parser.add_argument('--epochs', default=16, type=int)
parser.add_argument('--iters', default=20, type=int)
parser.add_argument('--batch_size', default=32, type=int)
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(0)

class PSA(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats, config):
        super(PSA, self).__init__()

        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        self.word_embedding = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))
        self.hist_convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=config['maxlen'] - K + 1) for K in config['kernel_sizes']])
        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU()
        
        self.train_records = config['train_records']
        self.test_records = config['test_records']


        self.hist_out = nn.Linear(300, 64)
        self.fc_out = nn.Sequential(
            nn.Linear(in_feats, 2 * hid_feats),
            nn.ReLU(),
            nn.Linear(2 * hid_feats, hid_feats)
        )

        print(self)


    def hist_aggregation(self, uids, records):
        uids = uids.cpu().numpy()
        hist_list = []
        for u in uids:
            hist = torch.LongTensor(np.array(records[u])).to(device)
            hist_rep = self.word_embedding(hist)
            hist_rep = torch.mean(hist_rep, dim=0, keepdim=True)
            hist_list.append(hist_rep)
        hist = torch.cat(hist_list, dim=0)
        return hist


    def records_aggregation(self, X_records):
        X_records = X_records.permute(0, 2, 1)
        conv_block = []
        for Conv, max_pooling in zip(self.hist_convs, self.max_poolings):
            act = self.relu(Conv(X_records))
            pool = max_pooling(act).squeeze(-1)
            conv_block.append(pool)
      
        features = torch.cat(conv_block, dim=1)
        features = self.hist_out(features)
        features = self.dropout(features)
        return features


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        tid, uid = data.tid, data.uid

        X_records = None
        if self.training:
            # print("aggregated publisher posting records from training set")
            X_records = self.hist_aggregation(uid, self.train_records)
            X_records = self.records_aggregation(X_records)

        else:
            # print("aggregated publisher posting records from test set")
            X_records = self.hist_aggregation(uid, self.test_records)
            X_records = self.records_aggregation(X_records)


        rootindex = data.rootindex
        x = self.fc_out(x[rootindex])
        x = self.dropout(x)
        x = X_records + x

        return x

    

class Net(nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats, config):
        super(Net, self).__init__()
        self.PSA = PSA(in_feats, hid_feats, out_feats, config)
        self.fc = nn.Linear(hid_feats,4)

            
    def forward(self, data):
        x_aggr = self.PSA(data)
        x = F.log_softmax(x_aggr, dim=1)
        return x


def train_psa(args, treeDic, x_test, x_train,TDdroprate, BUdroprate,lr, weight_decay, n_epochs, batchsize, dataname, iter):
    
    model = Net(5000,64,64, config).to(device)
    base_params = model.parameters()
    optimizer = torch.optim.Adam([{'params':base_params}], lr=lr, weight_decay=weight_decay)

    train_losses = []
    traindata_list, testdata_list = loadBiData_graph(dataname, treeDic, x_train, x_test, TDdroprate,BUdroprate)

    for epoch in range(n_epochs):

        model.train()
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, num_workers=5)

        avg_loss = []
        train_pred, y_train = [], []
        tqdm_train_loader = tqdm(train_loader)

        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            out_labels = model(Batch_data)
            loss = F.nll_loss(out_labels,Batch_data.y)
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            train_pred.append(pred)
            y_train.append(Batch_data.y)

        train_pred = torch.cat(train_pred, dim=0)
        y_train = torch.cat(y_train, dim=0)        

        correct = train_pred.eq(y_train).sum().item()
        train_acc = correct / len(y_train)
        print("Iter {:03d} | Epoch {:05d} | Train Acc. {:.4f}".format(iter, epoch, train_acc))
        train_losses.append(np.mean(avg_loss))
 
        if epoch == n_epochs - 1:
            model.eval()
            y_pred, y_test = [], []
            tqdm_test_loader = tqdm(test_loader)

            for Batch_data in tqdm_test_loader:
                Batch_data.to(device)
                val_out = model(Batch_data)
                _, val_pred = val_out.max(dim=1)

                y_pred.append(val_pred)
                y_test.append(Batch_data.y)

            y_pred = torch.cat(y_pred, dim=0)
            y_test = torch.cat(y_test, dim=0)        

            acc = accuracy_score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            precision, recall, fscore, _ = score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), average='macro')
    
    torch.save(model.state_dict(), 'checkpoint/' + args.model_name + '_' + dataname + '_iter' + str(iter) + '.m')

    print("-----------------End of Iter {:03d}-----------------".format(iter))
    print(['Global Test Accuracy:{:.4f}'.format(acc),
        'Precision:{:.4f}'.format(precision),
        'Recall:{:.4f}'.format(recall),
        'F1:{:.4f}'.format(fscore)])
        
    return acc, fscore


lr = 1e-4
weight_decay = 1e-5
if args.dataset_name == 'pheme_veracity_t10':
    lr = 0.0005
    weight_decay = 1e-4

batchsize = args.batch_size
n_epochs = args.epochs
TDdroprate = 0
BUdroprate = 0
datasetname=args.dataset_name 
iterations=args.iters
model="PSA"
device = torch.device('cuda')

test_accs = []
F1_all = []

word_embeddings = pickle.load(open('data/' + datasetname + '/word_embeddings.pkl', 'rb'))

# load publisher posting records 
train_records = pickle.load(open('data/' + datasetname + '/records_sep_train.pkl', 'rb'))
test_records = pickle.load(open('data/' + datasetname + '/records_sep_test.pkl', 'rb'))

config = {
    'embedding_weights': word_embeddings,
    'kernel_sizes':[3,4,5],
    'dropout':0.5,
    'maxlen':50,
    'train_records':train_records,
    'test_records':test_records
}

for iter in range(iterations):
    x_train, x_test = loadData_sep(datasetname)
    treeDic=loadTree(datasetname)
    acc, F1 = train_psa(args,
                        treeDic,
                        x_test,
                        x_train,
                        TDdroprate,BUdroprate,
                        lr, weight_decay,
                        n_epochs,
                        batchsize,
                        datasetname,
                        iter)

    test_accs.append(acc)
    F1_all.append(F1)

print("Total_Test_Accuracy: {:.4f}|Overall F1: {:.4f}".format(sum(test_accs) / iterations, sum(F1_all) /iterations))

with open('logs/log_' +  datasetname + '_' + args.model_name + '.' + 'iter' + str(iterations), 'a+') as f:
    f.write('All acc.s:{}\n'.format(test_accs))
    f.write('All F1.s (macro):{}\n'.format(F1_all))
    f.write('Average acc.: {} \n'.format(sum(test_accs) / iterations))
    f.write('Average F1.: {} \n'.format(sum(F1_all) / iterations))
    f.write('\n')