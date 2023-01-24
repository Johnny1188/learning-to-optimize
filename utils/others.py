import functools
import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()


def load_l2o_opter_ckpt(path, optee_cls, opter_cls, optee_config=None, opter_config=None):
    ckpt = torch.load(path)
    
    optee = w(optee_cls(**optee_config if optee_config is not None else {}))
    optee.load_state_dict(ckpt["optimizee"])
    optee_grads = ckpt["optimizee_grads"]
    for k, v in optee.named_parameters():
        v.grad = optee_grads[k]
    optee_updates = ckpt["optimizee_updates"] # predicted by l2o optimizer
    
    # opter = OPTIMIZER_CLS(preproc=OPTIMIZER_PREPROC)
    opter = opter_cls(**opter_config if opter_config is not None else {})
    opter.load_state_dict(ckpt["optimizer"])
    
    loss_history = ckpt["loss_history"]
    
    return optee, opter, optee_grads, optee_updates, loss_history


def load_baseline_opter_ckpt(path, optee_cls, opter_cls, optee_config=None, opter_config=None):
    ckpt = torch.load(path)
    
    optee = w(optee_cls(**optee_config if optee_config is not None else {}))
    optee.load_state_dict(ckpt["optimizee"])
    optee_grads = ckpt["optimizee_grads"]
    for k, v in optee.named_parameters():
        v.grad = optee_grads[k]
    
    opter = opter_cls(optee.parameters(), **opter_config if opter_config is not None else {})
    opter.load_state_dict(ckpt["optimizer"])
    
    loss_history = ckpt["loss_history"]
    
    return optee, opter, optee_grads, loss_history


def w(v):
    if USE_CUDA:
        return v.cuda()
    return v

def detach_var(v):
    var = w(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_grads(model):    
    for n,p in model.named_parameters():
        print(n, list(p.shape), f": {p.grad.abs().mean() if p.grad is not None else '---'}")
