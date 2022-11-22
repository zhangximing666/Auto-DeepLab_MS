"""Dynamic learning rate"""


class WarmUpPolyDecayLR:
    """Poly Learning Rate Scheduler with Warm Up"""
    def __init__(self, warmup_start_lr, base_lr, min_lr, warmup_iters, max_iteration):

        print('Using Poly LR Scheduler!')
        self.lr = base_lr
        self.max_iteration = max_iteration

        self.warmup_iters = warmup_iters
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_factor = (self.lr / warmup_start_lr) ** (1. / warmup_iters)

    def __call__(self, iteration):
        if self.warmup_iters > 0 and iteration < self.warmup_iters:
            lr = self.warmup_start_lr * (self.warmup_factor ** iteration)
        else:
            lr = self.lr * pow((1 - (iteration - self.warmup_iters) / (self.max_iteration - self.warmup_iters)), 0.9)
        return max(lr, self.min_lr)


def warmup_poly_lr(warmup_start_lr, base_lr, min_lr, warmup_iters, max_iteration):
    """List of warmup poly lr"""
    lr_scheduler = WarmUpPolyDecayLR(warmup_start_lr, base_lr, min_lr, warmup_iters, max_iteration)
    lr = []
    for i in range(max_iteration):
        lr.append(lr_scheduler(i))
    return lr
