from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from utils.cuda import Variable
import numpy as np
import torch
import json
import logging
import math

def read_config(file_path):
    json_object = json.load(open(file_path, 'r'))
    return json_object

def loggerConfig(log_file, verbose=2):
   logger      = logging.getLogger()
   formatter   = logging.Formatter('[%(levelname)-8s] (%(processName)-11s) %(message)s')
   fileHandler = logging.FileHandler(log_file, 'w')
   fileHandler.setFormatter(formatter)
   logger.addHandler(fileHandler)
   if verbose >= 2:
       logger.setLevel(logging.DEBUG)
   elif verbose >= 1:
       logger.setLevel(logging.INFO)
   else:
       logger.setLevel(logging.WARNING)
   return logger

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    # out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))             # 0.1.12
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out)) # 0.2.0
    return out
    
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)  

def normal(x, mu, sigma):
    pi = np.array([math.pi])
    pi = torch.from_numpy(pi).float()
    pi = Variable(pi)
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b