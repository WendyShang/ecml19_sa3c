import torch
import torch.nn as nn
import torch.nn.functional as F

class crelu(nn.Module):
    def __init__(self):
        '''
        crelu activation, crelu(x) = (relu(x), relu(-x))
        '''
        super(crelu, self).__init__()
    def forward(self, input):
        return torch.cat((F.relu(input), F.relu(input*-1)),1)
