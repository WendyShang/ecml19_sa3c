from torch.autograd import Variable
import torch
import torch.nn as nn

class sampler(nn.Module):
    def __init__(self, args):
        '''
        given mu and log(var), output samples using re-param trick
        '''
        super(sampler, self).__init__()
        self.cuda = args.cuda

    def forward(self, mu, logvar, eps=None):
        std = logvar.mul(0.5).exp_()
        if eps is None:
            if self.cuda:
                eps = torch.cuda.FloatTensor(std.size()).normal_()
            else:
                eps = torch.FloatTensor(std.size()).normal_()
            eps = Variable(eps)
        return eps.mul(std).add_(mu)
