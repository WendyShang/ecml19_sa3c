from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args):
        self.logger = args.logger

        super(Model, self).__init__()
        if args.gpu >= 0:
            self.use_cuda = True
        else:
            self.logger.warning("ERROR: Please use CUDA!")
            raise AssertionError()

        self.hidden_dim = args.hidden_dim
        self.dtype = args.dtype
        self.input_dims     = {}
        self.input_dims[0]  = args.hist_len   
        self.input_dims[1]  = args.state_shape
        self.output_dims    = args.action_dim
        self.preprocess_mode = args.preprocess_mode

    def _init_weights(self):
        raise NotImplementedError("not implemented in base calss")

    def print_model(self):
        self.logger.warning("<-----------------------------------> Model")
        self.logger.warning(self)

    def _reset(self):           # NOTE: should be called at each child's __init__
        self._init_weights()
        self.type(self.dtype)   # currently on support cuda float! 
        self.print_model()

    def forward(self, input):
        raise NotImplementedError("not implemented in base calss")
