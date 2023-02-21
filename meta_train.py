import copy
import json
import os.path
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from data import MNIST
from meta_module import *
from optimizee import (
    MNISTLeakyRelu,
    MNISTNet,
    MNISTNet2Layer,
    MNISTNetBig,
    MNISTRelu,
    MNISTReluBatchNorm,
    MNISTSimoidBatchNorm,
)
from optimizer import Optimizer
from regularization import (
    regularize_updates_constraints,
    regularize_updates_rescale_constraints,
    regularize_updates_scale_constraints,
    regularize_updates_translation_constraints,
)
from training import fit_optimizer
from utils.others import count_parameters, detach_var, print_grads, w
from utils.visualization import get_model_dot


def get_config():
    config = {  # global config
        "opter_cls": Optimizer,
        "opter_config": {
            "preproc": True,
            "additional_inp_dim": 0,
            "manual_init_output_params": False,
        },
        "eval_n_tests": 10,
        "ckpt_base_dir": f"./ckpt/{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}",
        "ckpt_baselines_dir": "./ckpt/baselines",
    }

    config["meta_training"] = {  # training the optimizer
        "data_cls": MNIST,
        "optee_cls": MNISTReluBatchNorm,
        "optee_config": {"affine": True, "track_running_stats": False},
        "n_epochs": 50,
        "n_optim_runs_per_epoch": 20,
        "n_iters": 200,
        "unroll": 20,
        "n_tests": 1,
        "optee_updates_lr": 0.1,
        "opter_lr": 0.01,
        "log_unroll_losses": True,
        "opter_updates_reg_func": regularize_updates_constraints,
        "reg_mul": 0.001,
        "eval_iter_freq": 10,
        "ckpt_iter_freq": 5,
        "ckpt_dir": None,  # will be set later
    }

    config["meta_testing"] = {  # testing the optimizer
        "data_cls": config["meta_training"]["data_cls"],
        "optee_cls": config["meta_training"]["optee_cls"],
        "optee_config": config["meta_training"]["optee_config"],
        "unroll": 1,
        "n_iters": 500,
        "optee_updates_lr": config["meta_training"]["optee_updates_lr"],
        "train_opter": False,
        "opter_optim": None,
        "ckpt_iter_freq": 5,
        "ckpt_dir": None,  # will be set later
    }

    ### additional config
    config[
        "ckpt_base_dir"
    ] = f"{config['ckpt_base_dir']}_{config['meta_training']['optee_cls'].__name__}_{config['opter_cls'].__name__}"
    # config["ckpt_baselines_dir"] = os.path.join(config["ckpt_base_dir"], "baselines")
    config["meta_training"]["ckpt_dir"] = os.path.join(
        config["ckpt_base_dir"], "meta_training"
    )
    config["meta_testing"]["ckpt_dir"] = os.path.join(
        config["ckpt_base_dir"], "meta_testing"
    )

    return config


def meta_train(config, seed=0):
    ### prepare ckpt dirs
    os.makedirs(config["ckpt_base_dir"], exist_ok=True)
    os.makedirs(config["meta_training"]["ckpt_dir"], exist_ok=True)
    os.makedirs(config["meta_testing"]["ckpt_dir"], exist_ok=True)
    os.makedirs(config["ckpt_baselines_dir"], exist_ok=True)

    ### dump config
    print(
        "-----\nMeta-training starts\nConfig:\n",
        json.dumps(config, indent=4, default=str),
    )
    with open(os.path.join(config["ckpt_base_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)

    ### meta-train a new L2O optimizer model
    torch.manual_seed(seed)
    np.random.seed(seed)

    best_training_loss, metrics, opter_state_dict = fit_optimizer(
        opter_cls=config["opter_cls"],
        opter_config=config["opter_config"],
        **config["meta_training"],
    )

    print(
        f"-----\nMeta-training finished"
        f"\n\t Best training loss: {best_training_loss}"
    )

    ### save the final L2O optimizer
    path_to_save = os.path.join(config["ckpt_base_dir"], f"l2o_optimizer.pt")
    try:
        torch.save(
            {
                "state_dict": opter_state_dict,
                "config": config,
                "loss": best_training_loss,
                "metrics": metrics,
            },
            path_to_save,
        )
        print(f"\t Everything saved to {path_to_save}")
    except:
        ### try dumping at least the state dict
        torch.save(opter_state_dict, "l2o_optimizer_state_dict.pt")
        print(f"\t Model state dict saved to {path_to_save}")

    opter = w(
        config["opter_cls"](
            **config["opter_config"] if config["opter_config"] is not None else {}
        )
    )
    opter.load_state_dict(opter_state_dict)

    return opter


if __name__ == "__main__":
    config = get_config()
    meta_train(config, seed=0)
