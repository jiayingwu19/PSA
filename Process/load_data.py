import random
from random import shuffle
import os

cwd=os.getcwd()

def loadData_sep(obj):
    print("Separate events")

    trainPath = os.path.join(cwd,"data/" +obj+"/"+ obj + "_eventsep.train")
    testPath = os.path.join(cwd,"data/" +obj+"/"+ obj + "_eventsep.test")
    x_train, x_test = [], []
    with open(trainPath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            uid, sid, content, label = line.strip().split("\t")
            x_train.append(sid)
    f.close()

    with open(testPath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            uid, sid, content, label = line.strip().split("\t")
            x_test.append(sid)
    f.close()

    return x_train, x_test


a, b = loadData_sep('Twitter15')