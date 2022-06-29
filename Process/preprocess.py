# -*- coding: utf-8 -*-
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
import pickle 
import re
import itertools
from collections import Counter
import gensim
import scipy.sparse as sp

cwd=os.getcwd()
np.random.seed(0)
w2v_dim = 300

class Node_tweet(object):

    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None


def str2matrix(Str):  # str = index:wordfreq index:wordfreq

    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex


def constructMat(tree):

    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            rootindex=indexC-1
            root_index=nodeC.index
            root_word=nodeC.word
    rootfeat = np.zeros([1, 5000])
    if len(root_index)>0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    matrix=np.zeros([len(index2node),len(index2node)])
    row=[]
    col=[]
    x_word=[]
    x_index=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                row.append(index_i)
                col.append(index_j)
        x_word.append(index2node[index_i+1].word)
        x_index.append(index2node[index_i+1].index)
    edgematrix=[row,col]
    return x_word, x_index, edgematrix,rootfeat,rootindex


def getfeature(x_word,x_index):

    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i])>0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x


def clean_str_cut(string, task):

    string = re.sub(r"[^A-Za-z0-9(),!?#@\'\`]", " ", string)
    string = re.sub(r"\'m", " am", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " had", string)
    string = re.sub(r"\'ll", " will", string)

    string = re.sub(r"'", " ' ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"#", " # ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    words = string.strip().lower().split()
    return words


def vocab_to_word2vec(fname, vocab):

    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    count_missing = 0
    for word in vocab:
        if model.__contains__(word):
            word_vecs[word] = model[word]
        else:
            #add unknown words by generating random word vectors
            count_missing += 1
            word_vecs[word] = np.random.uniform(-0.25, 0.25, w2v_dim)

    print(str(len(word_vecs) - count_missing)+" words found in word2vec.")
    print(str(count_missing)+" words not found, generated by random.")
    return word_vecs


def build_vocab_word2vec(sentences, w2v_path='numberbatch-en.txt'):

    # Build vocabulary
    vocabulary_inv = []
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv += [x[0] for x in word_counts.most_common() if x[1] >= 2]  #
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    print("embedding_weights generation.......")
    word2vec = vocab_to_word2vec(w2v_path, vocabulary)     #
    embedding_weights = build_word_embedding_weights(word2vec, vocabulary_inv)
    return vocabulary, embedding_weights


def pad_sequence(X, max_len=50):
    X_pad = []
    len_docs = []
    for doc in X:
        len_docs.append(len(doc))
        if len(doc) >= max_len:
            doc = doc[:max_len]
        else:
            doc = [0] * (max_len - len(doc)) + doc
        X_pad.append(doc)
    return X_pad, len_docs


def build_word_embedding_weights(word_vecs, vocabulary_inv):

    vocab_size = len(vocabulary_inv)
    embedding_weights = np.zeros(shape=(vocab_size+1, w2v_dim), dtype='float32')
    #initialize the first row
    embedding_weights[0] = np.zeros(shape=(w2v_dim,) )

    for idx in range(1, vocab_size):
        embedding_weights[idx] = word_vecs[vocabulary_inv[idx]]
    print("Embedding matrix of size " + str(np.shape(embedding_weights)))
    return embedding_weights


def build_input_data(X, eids, vocabulary):

    x = [[vocabulary[word] for word in sentence if word in vocabulary] for sentence in X]
    x, ld = pad_sequence(x)

    wid_dict = dict(zip(eids, x))

    return x, ld, wid_dict


def aggr_hist(wids, uids):
    hist = list(zip(uids, wids))
    cnt_hist = []
    for k in Counter(uids).keys():
        temp = []
        for pair in hist:
            if pair[0] == k:
                temp.append(pair[1])
        cnt_hist.append([k, temp])
    
    cnt_hist = dict((i,j) for [i,j] in cnt_hist)

    return cnt_hist


def get_splits_records(dataPath, wid_dict):
    X_wids, uids = [], []
    for line in open(dataPath):
        line = line.rstrip()
        uid, eid = line.strip().split("\t")[0], line.strip().split("\t")[1]
        X_wids.append(wid_dict[eid])
        uids.append(int(uid))
    
    records = aggr_hist(np.array(X_wids), np.array(uids))
    
    return records


def collect_train_test_hist(obj, wid_dict):

    # collection of publisher posting records
    # separately collect publisher records from the training and test data

    sep_train_path = os.path.join(cwd,"data/" +obj+"/"+ obj + "_eventsep.train")
    sep_test_path = os.path.join(cwd,"data/" +obj+"/"+ obj + "_eventsep.test")
    
    sep_train_records = get_splits_records(sep_train_path, wid_dict)
    sep_test_records = get_splits_records(sep_test_path, wid_dict)

    pickle.dump(sep_train_records, open(os.path.join(cwd, "data/" + obj + "/records_sep_train.pkl"), 'wb'))
    pickle.dump(sep_test_records, open(os.path.join(cwd, "data/" + obj + "/records_sep_test.pkl"), 'wb'))

    print("records of train/test splits written to file")


def main(obj):
    treePath = os.path.join(cwd, 'data/' + obj + '/data.TD_RvNN.vol_5000.txt')
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]

        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}

    labelPath = os.path.join(cwd, "data/" + obj + "/" + obj + ".txt")
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']

    print("Dataset name: ", obj)
    print("loading tree label")
    event, y = [], []
    X_wids, eids, uids = [], [], []
    l1 = l2 = l3 = l4 = 0
    labelDic = {}
    tidDic = {}

    for line in open(labelPath):
        line = line.rstrip()

        uid, eid, content, label = line.strip().split("\t")
        eids.append(eid)
        uids.append(int(uid))
        tidDic[eid] = uid
        X_wids.append(clean_str_cut(content, obj))
        label=label.lower()
        event.append(eid)
        if label in labelset_nonR:
            labelDic[eid]=0
            l1 += 1
        if label  in labelset_f:
            labelDic[eid]=1
            l2 += 1
        if label  in labelset_t:
            labelDic[eid]=2
            l3 += 1
        if label  in labelset_u:
            labelDic[eid]=3
            l4 += 1
    print("Number of instances: ", l1 + l2 + l3 + l4)
    print("NR: ", l1, "\nFR: ",l2, "\nTR: ", l3,"\nUR: ", l4)
    
    uids = np.array(uids)

    # vocabulary, word_embeddings = build_vocab_word2vec(X_wids, 'data/twitter_w2v.bin')
    # X_source_wid, ld_x, wid_dict = build_input_data(X_wids, eids, vocabulary)
    # collect_train_test_hist(obj, wid_dict)
    # vocabPath = os.path.join(cwd, "data/" + obj + "/vocab.pkl")
    # embedPath = os.path.join(cwd, "data/" + obj + "/word_embeddings.pkl")
    # pickle.dump(vocabulary, open(vocabPath, 'wb'))
    # pickle.dump(word_embeddings, open(embedPath, 'wb'))


    def loadEid(event,id,y,obj):
        if event is None:
            return None
        if len(event) < 2:
            return None
        if len(event)>1:
            x_word, x_index, tree, rootfeat, rootindex = constructMat(event)
            x_x = getfeature(x_word, x_index)
            tid = int(id)           
            uid = int(tidDic[id])
            rootfeat, tree, x_x, rootindex, y, tid, uid = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(
                rootindex), np.array(y), np.array(tid), np.array(uid)
            np.savez( os.path.join(cwd, 'data/'+obj+'_graph/'+id+'.npz'), x=x_x,root=rootfeat,edgeindex=tree,rootindex=rootindex,y=y, \
                        tid=tid, uid=uid)
            return None
    print("loading dataset")
    Parallel(n_jobs=30, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid], obj) \
            for eid in tqdm(event))
    return

if __name__ == '__main__':
    obj= sys.argv[1]
    main(obj)