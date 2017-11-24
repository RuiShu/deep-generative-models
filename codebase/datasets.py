import numpy as np
import os
from scipy.io import loadmat
import scipy
import sys
import cPickle as pkl
import tensorbayes as tb
from itertools import izip

def u2t(x):
    """
    Convert uint-8 encoding to 'tanh' encoding (aka range [-1, 1])
    """
    return x.astype('float32') / 255 * 2 - 1

def s2t(x):
    """
    Convert 'sigmoid' encoding (aka range [0, 1]) to 'tanh' encoding
    """
    return x * 2 - 1

def create_labeled_data(x, y, seed, npc):
    print "Create labeled data, npc:", npc
    state = np.random.get_state()
    np.random.seed(seed)
    shuffle = np.random.permutation(len(x))
    x, y = x[shuffle], y[shuffle]
    np.random.set_state(state)

    x_l, y_l, i_l = [], [], []
    for k in xrange(10):
        idx = y.argmax(-1) == k
        x_l += [x[idx][:npc]]
        y_l += [y[idx][:npc]]
    x_l = np.concatenate(x_l, axis=0)
    y_l = np.concatenate(y_l, axis=0)
    return x_l, y_l

class Data(object):
    def __init__(self, images, labels=None, labeler=None, cast=False):
        self.images = images
        self.labels = labels
        self.labeler = labeler
        self.cast = cast

    def preprocess(self, x):
        if self.cast:
            return u2t(x)
        else:
            return x

    def next_batch(self, bs):
        idx = np.random.choice(len(self.images), bs, replace=False)
        x = self.preprocess(self.images[idx])
        y = self.labeler(x) if self.labels is None else self.labels[idx]
        return x, y

class Svhn(object):
    def __init__(self, setting='train', seed=0, npc=None):
        print "Loading SVHN"
        sys.stdout.flush()
        path = 'data'
        train = loadmat(os.path.join(path, 'train_32x32.mat'))
        test = loadmat(os.path.join(path, 'test_32x32.mat'))

        # Change format
        trainx, trainy = self.change_format(train)
        testx, testy = self.change_format(test)

        # Convert to one-hot
        trainy = np.eye(10)[trainy]
        testy = np.eye(10)[testy]

        # Filter via npc if not None
        if npc:
            trainx, trainy = create_labeled_data(trainx, trainy, seed, npc)

        if setting == 'extra':
            print "Adding extra data"
            extra = loadmat(os.path.join(path, 'extra_32x32.mat'))
            extrax, extray = self.change_format(extra)
            extray = np.eye(10)[extray]
            trainx = np.concatenate((trainx, extrax), axis=0)
            trainy = np.concatenate((trainy, extray), axis=0)

        self.train = Data(trainx, trainy, cast=True)
        self.test = Data(testx, testy, cast=True)

    @staticmethod
    def change_format(mat):
        x = mat['X'].transpose((3, 0, 1, 2))
        y = mat['y'].reshape(-1)
        y[y == 10] = 0
        return x, y
