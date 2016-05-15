import gzip
import pickle
from urllib.request import urlretrieve
import os
import numpy as np

from os.path import isfile

PREFIX = os.getenv('ATISDATA', '')

def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out  = [l[:i] for i in range(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in range(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win//2 * [-1] + l + win//2 * [-1]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

def download(origin):
    '''
    download the corresponding atis file
    from http://www-etud.iro.umontreal.ca/~mesnilgr/atis/
    '''
    print('Downloading data from %s' % origin)
    name = origin.split('/')[-1]
    #urllib.request.urlretrieve(origin, name)
    urlretrieve(origin, name)


def download_dropbox():
    '''
    download from drop box in the meantime
    '''
    print('Downloading data from https://www.dropbox.com/s/3lxl9jsbw0j7h8a/atis.pkl?dl=0')
    os.system('wget -O atis.pkl https://www.dropbox.com/s/3lxl9jsbw0j7h8a/atis.pkl?dl=0')

def load_dropbox(filename):
    if not isfile(filename):
        #download('http://www-etud.iro.umontreal.ca/~mesnilgr/atis/'+filename)
        download_dropbox()
    #f = gzip.open(filename,'rb')
    f = open(filename,'rb')
    return f

def load_udem(filename):
    if not isfile(filename):
        download('http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/'+filename)
    f = gzip.open(filename,'rb')
    return f


def atisfull():
    f = load_dropbox(PREFIX + 'atis.pkl')
    train_set, test_set, dicts = pickle.load(f, encoding='latin1')
    return train_set, test_set, dicts

def atisfold(fold):
    assert fold in range(5)
    f = load_udem(PREFIX + 'atis.fold'+str(fold)+'.pkl.gz')
    train_set, valid_set, test_set, dicts = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set, dicts

if __name__ == '__main__':

    ''' visualize a few sentences '''

    import pdb

    w2ne, w2la = {}, {}
    train, test, dic = atisfull()
    train, _, test, dic = atisfold(1)

    w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']

    idx2w  = dict((v,k) for k,v in w2idx.items())
    idx2ne = dict((v,k) for k,v in ne2idx.items())
    idx2la = dict((v,k) for k,v in labels2idx.items())

    test_x,  test_ne,  test_label  = test
    train_x, train_ne, train_label = train
    wlength = 35

    for i in range(len(train_x)):
        cwords = contextwin(train_x[i], 7)
        batch = minibatch(cwords, 9)
        words = map(lambda x: np.asarray(x).astype('int32'), batch)
        labels = train_label[i]
        for word_batch , label_last_word in zip(words, labels):
            print('word batch:', word_batch)
            print('labels', label_last_word)




    for e in ['train','test']:
      for sw, se, sl in zip(eval(e+'_x'), eval(e+'_ne'), eval(e+'_label')):
        print ('WORD'.rjust(wlength), 'LABEL'.rjust(wlength), '?'.rjust(wlength), 'WORD INDEX'.rjust(wlength), 'LABEL INDEX'.rjust(wlength))
        for wx, la, ne in zip(sw, sl, se): print(idx2w[wx].rjust(wlength), idx2la[la].rjust(wlength), str(ne).rjust(wlength),
                                                 str(wx).rjust(wlength), str(la).rjust(wlength))
        print('\n'+'**'*30+'\n')
        pdb.set_trace()
