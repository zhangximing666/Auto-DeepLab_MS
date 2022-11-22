"""Batch normalization."""
import mindspore.nn as nn


class NormReLU(nn.Cell):
    """ABN module, activation function: nn.ReLU"""
    def __init__(self, num_features, momentum, eps, affine=True, parallel=True):
        super(NormReLU, self).__init__()
        self.op = nn.SequentialCell(
            BatchNormalization(num_features, momentum, eps, affine=affine, parallel=parallel),
            nn.ReLU()
        )

    def construct(self, x):
        """construct"""
        return self.op(x)


class NormLeakyReLU(nn.Cell):
    """ABN module, activation function: nn.LeakyReLU"""
    def __init__(self, num_features, momentum, eps, slope=0.01, affine=True, parallel=True):
        super(NormLeakyReLU, self).__init__()

        self.op = nn.SequentialCell(
            BatchNormalization(num_features, momentum, eps, affine=affine, parallel=parallel),
            nn.LeakyReLU(slope)
        )

    def construct(self, x):
        """construct"""
        return self.op(x)


class BatchNormalization(nn.Cell):
    """batch normalization"""
    def __init__(self, num_features, momentum=0.9, eps=1e-5, affine=True, parallel=True):
        super(BatchNormalization, self).__init__()
        if parallel:
            self.op = nn.SyncBatchNorm(num_features, eps, momentum, affine=affine)
        else:
            self.op = nn.BatchNorm2d(num_features, eps, momentum, affine=affine)

    def construct(self, x):
        """construct"""
        return self.op(x)
