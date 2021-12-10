import numpy as np


def _hist(predict, label, n):
    """
    _hist
    inputs:
        - predict (ndarray)
        - label (ndarray)
        - n (int) - number of classes
    outputs:
        - fast histogram
    """
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(np.int32) + predict[k], minlength=n ** 2).reshape(n, n)


class CityscapesDataLoader(object):
    def __init__(self):
        super(CityscapesDataLoader, self).__init__()
        
