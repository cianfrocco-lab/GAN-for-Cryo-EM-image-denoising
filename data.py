from config import Config as conf

import os
import numpy as np
import random
import scipy.misc

def load_train(path):
    path1=path+"/train/original"
    path2=path+"/train/noisy"
    for i in random.sample(os.listdir(path1),len(os.listdir(path1))):
        file_path1 = os.path.join(path1, i)
        file_path2 = os.path.join(path2, i)
        all1 = scipy.misc.imread(file_path1)
        all2 = scipy.misc.imread(file_path2)
        img=all1
        cond=all2
        yield (img, cond, i)


def load_test(path):
    path1=path+"/test/original"
    path2=path+"/test/noisy"
    for i in os.listdir(path1):
        file_path1 = os.path.join(path1, i)
        file_path2 = os.path.join(path2, i)
        all1 = scipy.misc.imread(file_path1)
        all2 = scipy.misc.imread(file_path2)
        img=all1
        cond=all2
        yield (img, cond, i)

def load_data():
    data = dict()
    data["train"] = lambda: load_train(conf.data_path_train )
    data["test"] = lambda: load_test(conf.data_path_test )
    return data
