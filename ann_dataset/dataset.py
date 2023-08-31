import os
import mmap
import h5py
import numpy as np
from sklearn import preprocessing


def mmap_read(dataset, path):
    offset = dataset.id.get_offset()
    shape = dataset.shape
    dtype = dataset.dtype
    file = open(path, 'rb')
    fileno = file.fileno()
    mapping = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
    return np.frombuffer(mapping, dtype=dtype, count=np.prod(shape), offset=offset).reshape(shape)


class Dataset:
    def __init__(self):
        self.name = ""
        self.metric = "L2"
        self.d = -1
        self.nb = -1
        self.nq = -1
        self.base = None
        self.query = None
        self.gt = None
        self.file = None
        self.path = ""
        self.normalize = True

    def evaluate(self, pred, k=None):
        nq, topk = pred.shape
        if k is not None:
            topk = k
        gt = self.get_groundtruth(topk)
        cnt = 0
        for i in range(nq):
            cnt += np.intersect1d(pred[i], gt[i]).size
        return cnt / self.nq / k

    def get_database(self):
        ret = mmap_read(self.file['train'], self.path)
        if self.metric == "IP" and self.normalize:
            ret = preprocessing.normalize(ret)
        return ret

    def get_queries(self):
        ret = np.array(self.file['test'])
        if self.metric == "IP" and self.normalize:
            ret = preprocessing.normalize(ret)
        return ret

    def get_groundtruth(self, k):
        ret = np.array(self.file['neighbors'])
        return ret[:, :k]

    def get_fname(self, dir):
        if dir is None:
            dir = "datasets"
        if not os.path.exists(dir):
            os.mkdir(dir)
        return f'{dir}/{self.name}.hdf5'


def download(name):
    url = f'https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/{name}.hdf5'
    return url


class DatasetSIFT1M(Dataset):
    name = "sift-128-euclidean"
    metric = "L2"

    def __init__(self, dir=None):
        self.path = self.get_fname(dir)
        if not os.path.exists(self.path):
            os.system(
                f'wget --output-document {self.path} {download(self.name)}')
        self.file = h5py.File(self.path)


class DatasetFashionMnist(Dataset):
    name = "fashion-mnist-784-euclidean"
    metric = "L2"

    def __init__(self, dir=None):
        self.path = self.get_fname(dir)
        if not os.path.exists(self.path):
            os.system(
                f'wget --output-document {self.path} {download(self.name)}')
        self.file = h5py.File(self.path)


class DatasetNYTimes(Dataset):
    name = "nytimes-256-angular"
    metric = "IP"
    normalize = True

    def __init__(self, dir=None):
        self.path = self.get_fname(dir)
        if not os.path.exists(self.path):
            os.system(
                f'wget --output-document {self.path} {download(self.name)}')
        self.file = h5py.File(self.path)


class DatasetGlove100(Dataset):
    name = "glove-100-angular"
    metric = "IP"
    normalize = True

    def __init__(self, dir=None):
        self.path = self.get_fname(dir)
        if not os.path.exists(self.path):
            os.system(
                f'wget --output-document {self.path} {download(self.name)}')
        self.file = h5py.File(self.path)


class DatasetGlove25(Dataset):
    name = "glove-25-angular"
    metric = "IP"
    normalize = True

    def __init__(self, dir=None):
        self.path = self.get_fname(dir)
        if not os.path.exists(self.path):
            os.system(
                f'wget --output-document {self.path} {download(self.name)}')
        self.file = h5py.File(self.path)


class DatasetGlove200(Dataset):
    name = "glove-200-angular"
    metric = "IP"
    normalize = True

    def __init__(self, dir=None):
        self.path = self.get_fname(dir)
        if not os.path.exists(self.path):
            os.system(
                f'wget --output-document {self.path} {download(self.name)}')
        self.file = h5py.File(self.path)


class DatasetLastFM64(Dataset):
    name = "lastfm-64-dot"
    metric = "IP"
    normalize = True

    def __init__(self, dir=None):
        self.path = self.get_fname(dir)
        if not os.path.exists(self.path):
            os.system(
                f'wget --output-document {self.path} {download(self.name)}')
        self.file = h5py.File(self.path)


class DatasetGIST960(Dataset):
    name = "gist-960-euclidean"
    metric = "L2"

    def __init__(self, dir=None):
        self.path = self.get_fname(dir)
        if not os.path.exists(self.path):
            os.system(
                f'wget --output-document {self.path} {download(self.name)}')
        self.file = h5py.File(self.path)


class DatasetText2Image10M(Dataset):
    name = "text2image-10M"
    metric = "IP"
    normalize = False

    def __init__(self, dir=None):
        self.path = self.get_fname(dir)
        if not os.path.exists(self.path):
            os.system(
                f'wget --output-document {self.path} {download(self.name)}')
        self.file = h5py.File(self.path)


class DatasetCohere(Dataset):
    name = "cohere-768-angular"
    metric = "IP"
    normalize = True

    def __init__(self, dir=None):
        self.path = self.get_fname(dir)
        if not os.path.exists(self.path):
            os.system(
                f'wget --output-document {self.path} {download(self.name)}')
        self.file = h5py.File(self.path)


dataset_dict = {'sift-128-euclidean': DatasetSIFT1M, 'fashion-mnist-784-euclidean': DatasetFashionMnist,
                'nytimes-256-angular': DatasetNYTimes, 'glove-100-angular': DatasetGlove100,
                'glove-25-angular': DatasetGlove25, 'glove-200-angular': DatasetGlove200, 'lastfm-64-dot': DatasetLastFM64,
                'gist-960-euclidean': DatasetGIST960, 'cohere-768-angular': DatasetCohere, 'text2image-10M': DatasetText2Image10M}


def list_datasets():
    return list(dataset_dict.keys())


def to_fbin(x, fname):
    f = open(fname, "wb")
    n, d = x.shape
    np.array([n, d], dtype='uint32').tofile(f)
    x.tofile(f)
