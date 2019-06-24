import torch.autograd as autograd

# by default, we always use GPU
USE_CUDA = True

def Variable(data, *args, **kwargs):
    var = autograd.Variable(data, *args, **kwargs)
    if USE_CUDA:
        var = var.cuda()
    return var

    
