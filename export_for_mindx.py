"""export checkpoint file into mindir models"""
import numpy as np

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

from src.core.model import AutoDeepLab
from src.config import obtain_autodeeplab_args
from src.utils.utils import InferWithFlipNetwork

context.set_context(mode=context.GRAPH_MODE, save_graphs=False)

if __name__ == "__main__":
    args = obtain_autodeeplab_args()
    args.total_iters = 0

    # net
    autodeeplab = AutoDeepLab(args)

    # load checkpoint
    param_dict = load_checkpoint(args.ckpt_name)

    # load the parameter into net
    load_param_into_net(autodeeplab, param_dict)
    network = InferWithFlipNetwork(autodeeplab, input_format=args.input_format)

    input_data = np.random.uniform(0.0, 1.0, size=[1, 1024, 2048, 3]).astype(np.float32)
    export(network, Tensor(input_data), file_name='Auto-DeepLab-s', file_format=args.file_format)
