import numpy
import random
import elman
import load

if __name__ == '__main__':
    fold = 3
    lr = 0.06
    win = 7
    bsteps = 9
    nhidden = 100
    emb_dim = 100
    nepochs = 10
    nbatch = 1000
    lmbda = 0.01
    seed = 345

    # load the dataset
    train_set, valid_set, test_set, dic = load.atisfold(fold)
    idx2label = dict((k,v) for v,k in dic['labels2idx'].items())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].items())

    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex,  test_ne,  test_y  = test_set

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)

    # instantiate the model
    numpy.random.seed(seed)
    random.seed(seed)

    network = elman.Network(nhidden, nclasses, vocsize, emb_dim, win)
    network.train(train_lex, train_y, valid_lex, valid_y, lr, nepochs, bsteps)
