import numpy as np
from PIL import Image


cityspallete = [
        128, 64, 128,
        244, 35, 232,
        70, 70, 70,
        102, 102, 156,
        190, 153, 153,
        153, 153, 153,
        250, 170, 30,
        220, 220, 0,
        107, 142, 35,
        152, 251, 152,
        0, 130, 180,
        220, 20, 60,
        255, 0, 0,
        0, 0, 142,
        0, 0, 70,
        0, 60, 100,
        0, 80, 100,
        0, 0, 230,
        119, 11, 32,
    ]


def fast_hist(predict, label, n):
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


def label_to_color_image(npimg) -> Image:
    img = Image.fromarray(npimg.astype('uint8'), "P")
    img.putpalette(cityspallete)
    out_img = img.convert('RGB')
    return out_img