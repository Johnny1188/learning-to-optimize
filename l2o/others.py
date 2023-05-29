import functools
import os

import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()


def load_ckpt(dir_path):
    ### load pretrained L2O optimizer model and config
    ckpt = torch.load(os.path.join(dir_path, "l2o_optimizer.pt"))
    config = ckpt["config"]

    opter = w(
        config["opter_cls"](
            **config["opter_config"] if config["opter_config"] is not None else {}
        )
    )
    opter.load_state_dict(ckpt["state_dict"])

    return opter, config, ckpt


def load_l2o_opter_ckpt(
    path, optee_cls, opter_cls, optee_config=None, opter_config=None
):
    ckpt = torch.load(path)

    optee = w(optee_cls(**optee_config if optee_config is not None else {}))
    optee.load_state_dict(ckpt["optimizee"])
    optee_grads = ckpt["optimizee_grads"]
    for k, v in optee.named_parameters():
        v.grad = optee_grads[k]
    optee_updates = ckpt["optimizee_updates"]  # predicted by l2o optimizer

    opter = opter_cls(**opter_config if opter_config is not None else {})
    opter.load_state_dict(ckpt["optimizer"])

    metrics = ckpt["metrics"]

    return optee, opter, optee_grads, optee_updates, metrics


def load_baseline_opter_ckpt(
    path, optee_cls, opter_cls, optee_config=None, opter_config=None
):
    ckpt = torch.load(path)

    optee = w(optee_cls(**optee_config if optee_config is not None else {}))
    optee.load_state_dict(ckpt["optimizee"])
    optee_grads = ckpt["optimizee_grads"]
    for k, v in optee.named_parameters():
        v.grad = optee_grads[k]

    if "lr" in opter_config and type(opter_config["lr"]) not in (int, float):
        opter_config = opter_config.copy()
        opter_config["lr"] = ckpt["optimizer"]["param_groups"][0]["lr"]

    opter = opter_cls(
        optee.parameters(), **opter_config if opter_config is not None else {}
    )
    opter.load_state_dict(ckpt["optimizer"])

    metrics = ckpt["metrics"]

    return optee, opter, optee_grads, metrics


# def load_ckpt(
#     path, optee_cls, opter_cls, optee_config=None, opter_config=None, is_l2o=True
# ):
#     ckpt = torch.load(path)

#     optee = w(optee_cls(**optee_config if optee_config is not None else {}))
#     optee.load_state_dict(ckpt["optimizee"])
#     optee_grads = ckpt["optimizee_grads"]
#     for k, v in optee.named_parameters():
#         v.grad = optee_grads[k]
    
#     if is_l2o:
#         opter = opter_cls(**opter_config if opter_config is not None else {})
#         opter.load_state_dict(ckpt["optimizer"])
#         optee_updates = ckpt["optimizee_updates"]  # predicted by l2o optimizer
#     else:
#         opter = opter_cls(
#             optee.parameters(), **opter_config if opter_config is not None else {}
#         )
#         opter.load_state_dict(ckpt["optimizer"])
#         optee_updates = None

#     metrics = ckpt["metrics"]

#     return optee, opter, optee_grads, optee_updates, metrics


def dict_to_str(d):
    # inner_str = "_".join([f"{k}={v}" for k, v in d.items()])
    inner_str = ""
    for k, v in d.items():
        if callable(v):
            inner_str += f"{k}={v.__name__}_"
        else:
            inner_str += f"{k}={v}_"
    inner_str = inner_str[:-1]
    return "{" + inner_str + "}"


def get_baseline_ckpt_dir(
    opter_cls,
    opter_config,
    optee_cls,
    optee_config,
    data_cls,
    data_config,
):
    ckpt_dir = opter_cls.__name__
    ckpt_dir += "_"
    ckpt_dir += dict_to_str(opter_config)
    ckpt_dir += "_"
    ckpt_dir += optee_cls.__name__
    ckpt_dir += "_"
    ckpt_dir += dict_to_str(optee_config)
    ckpt_dir += "_"
    ckpt_dir += data_cls.__name__
    ckpt_dir += "_"
    ckpt_dir += dict_to_str(data_config)
    return ckpt_dir


def w(v):
    if USE_CUDA:
        return v.cuda()
    return v


def detach_var(v):
    var = w(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_grads(model):
    for n, p in model.named_parameters():
        print(
            n,
            list(p.shape),
            f": {p.grad.abs().mean() if p.grad is not None else '---'}",
        )
