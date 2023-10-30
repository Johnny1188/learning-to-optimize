import copy
import json
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from l2o.others import dict_to_str, get_baseline_ckpt_dir
from l2o.meta_module import *
from l2o.optimizer import Optimizer
from l2o.others import count_parameters, detach_var, rgetattr, rsetattr, w

from l2o.analysis import get_baseline_opter_param_updates
from l2o.lion_optimizer import Lion
from l2o.lion_optimizer import update_fn as lion_update_fn


def fit_normal(
    data_cls,
    optee_cls,
    opter_cls_list,
    n_tests=100,
    n_iters=100,
    data_config=None,
    optee_config=None,
    opter_configs=None,
    opter_mul_factors=None,
    eval_iter_freq=10,
    ckpt_iter_freq=None,
    ckpt_dir="",
    save_ckpts_for_all_test_runs=False,
):
    assert isinstance(opter_cls_list, list)
    assert isinstance(opter_configs, list) and len(opter_configs) == len(opter_cls_list)
    assert isinstance(opter_mul_factors, list) and len(opter_mul_factors) == len(opter_cls_list)

    if ckpt_iter_freq is not None:
        os.makedirs(ckpt_dir, exist_ok=True)

    metrics = {m: [] for m in ["train_loss", "train_acc", "test_loss", "test_acc"]}

    for test_i in range(n_tests):
        train_data = data_cls(
            training=True, **data_config if data_config is not None else {}
        )
        test_data = data_cls(
            training=False, **data_config if data_config is not None else {}
        )

        optee = w(optee_cls(**optee_config if optee_config is not None else {}))
        # opter = opter_cls(optee.parameters(), **opter_config)
        opters = []
        for opter_cls, opter_config in zip(opter_cls_list, opter_configs):
            opters.append(opter_cls(optee.parameters(), **opter_config))

        ### new test run
        for k in metrics.keys():
            metrics[k].append([])

        for iter_i in range(1, n_iters + 1):
            ### train
            optee.train()
            opters[0].zero_grad()
            train_loss, train_acc = optee(train_data, return_acc=True)
            train_loss.backward()

            ### save optee params before
            optee_params_before = {
                n: p.detach().clone() for n, p in optee.all_named_parameters()
            }

            ### update (after checkpointing)
            optee_update = dict()
            for opter, update_mul_factor in zip(opters, opter_mul_factors):
                curr_opter_update = get_baseline_opter_param_updates(optee, opter)
                for n in curr_opter_update.keys():
                    if n not in optee_update:
                        optee_update[n] = curr_opter_update[n] * update_mul_factor
                    else:
                        optee_update[n] += curr_opter_update[n] * update_mul_factor
                opter.step() # to update the internal state
            
            ### model checkpointing - save only for the first test run if save_ckpts_for_all_test_runs is False
            if ckpt_iter_freq and (iter_i % ckpt_iter_freq == 0 or iter_i == 1) \
                and (test_i == 0 or save_ckpts_for_all_test_runs):
                ckpt = {
                    "optimizee": optee.state_dict(),
                    "optimizee_grads": {
                        k: v.grad for k, v in optee.all_named_parameters()
                    },
                    "optimizee_updates": optee_update,
                    "optimizer": {opter.__class__.__name__: opter.state_dict() for opter in opters},
                    "metrics": metrics,
                }
                torch.save(ckpt, os.path.join(ckpt_dir, f"run{test_i}_{iter_i}.pt"))
            
            ### apply update
            for n, p in optee.all_named_parameters():
                # p.data.add_(optee_update[n])
                p.data = optee_params_before[n] + optee_update[n]

            ### log
            metrics["train_loss"][-1].append(train_loss.item())
            metrics["train_acc"][-1].append(train_acc.item())

            ### eval
            if eval_iter_freq and iter_i % eval_iter_freq == 0:
                optee.eval()
                test_loss, test_acc = optee(test_data, return_acc=True)
                metrics["test_loss"][-1].append(test_loss.item())
                metrics["test_acc"][-1].append(test_acc.item())
                optee.train()

    ### metrics to numpy arrays
    for k, v in metrics.items():
        metrics[k] = np.array(v)

    return metrics


def find_best_lr_normal(
    data_cls,
    optee_cls,
    opter_cls,
    n_tests=3,
    n_iters=50,
    optee_config=None,
    opter_config=None,
    consider_metric="train_loss",
    lrs_to_try=None,
):
    assert consider_metric in ["train_loss", "test_loss"]

    opter_config = copy.deepcopy(opter_config) if opter_config is not None else {}
    if lrs_to_try is None:
        lrs_to_try = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 3e0]

    best_loss_sum = np.inf
    best_lr = None

    for lr in lrs_to_try:
        opter_config["lr"] = lr
        metrics = fit_normal(
            data_cls=data_cls,
            optee_cls=optee_cls,
            opter_cls=opter_cls,
            n_tests=n_tests,
            n_iters=n_iters,
            optee_config=optee_config,
            opter_config=opter_config,
            eval_iter_freq=None if consider_metric == "train_loss" else 1,
            ckpt_iter_freq=None,
        )
        train_loss_sum = metrics[consider_metric].sum()
        if train_loss_sum < best_loss_sum:
            best_loss_sum = train_loss_sum
            best_lr = lr

    return best_lr


def meta_test_baselines(
    opter_combs,
    optees,
    config,
    use_existing_baselines=False,
    save_ckpts_for_all_test_runs=False
):
    """
    Parameters
    ----------
    baseline_opters : list of tuples
        List of tuples of the form (optimizer_name, optimizer_cls, optimizer_config).
    optees : list of tuples
        List of tuples of the form (optimizee_cls, optimizee_config).
    config : dict
        The config dictionary for the training run.
    use_existing_baselines : bool, optional
        Whether to use existing baselines if they exist, by default True.
    save_ckpts_for_all_test_runs : bool, optional
        Whether to save checkpoints for all test runs, by default False.
        
    Returns
    -------
    results : dict
        Dictionary of results.
    """
    assert not use_existing_baselines, "Not implemented"

    results = dict()
    for optee_cls, optee_config in optees:
        ### train optees with baseline optimizers (or load previous)
        for opter_names, opter_cls_list, opter_configs, opter_mul_factors in opter_combs:
            print(f"Training {optee_cls.__name__}_{dict_to_str(optee_config)} with {opter_names} optimizers (mul_factors: {opter_mul_factors})")
            torch.manual_seed(0)
            np.random.seed(0)

            ### set config
            run_config = deepcopy(config)
            run_config["meta_testing"]["optee_cls"] = optee_cls
            run_config["meta_testing"]["optee_config"] = optee_config

            ### prepare checkpointing
            if run_config["meta_testing"]["ckpt_iter_freq"] is not None:
                baseline_file_nickname = get_ckpt_dir(
                    opter_cls_list=opter_cls_list,
                    opter_configs=opter_configs,
                    opter_mul_factors=opter_mul_factors,
                    optee_cls=run_config["meta_testing"]["optee_cls"],
                    optee_config=run_config["meta_testing"]["optee_config"],
                    data_cls=run_config["meta_testing"]["data_cls"],
                    data_config=run_config["meta_testing"]["data_config"],
                )
                baseline_opter_dir = os.path.join(
                    os.environ["CKPT_PATH"], run_config["ckpt_baselines_dir"], baseline_file_nickname
                )
                os.makedirs(baseline_opter_dir, exist_ok=True)
                metrics_path = os.path.join(baseline_opter_dir, "metrics.npy")

            opter_configs_for_run = deepcopy(opter_configs)

            ### find best lr
            # if "lr" in opter_config_for_run and callable(opter_config_for_run["lr"]):
            #     print(f"  Finding best lr for {opter_name} optimizer")
            #     best_lr = opter_config_for_run["lr"](
            #         data_cls=run_config["meta_testing"]["data_cls"],
            #         optee_cls=run_config["meta_testing"]["optee_cls"],
            #         opter_cls=baseline_opter_cls,
            #         n_tests=3,
            #         n_iters=run_config["meta_testing"]["n_iters"] // 2,
            #         optee_config=run_config["meta_testing"]["optee_config"],
            #         opter_config=opter_config_for_run,
            #         consider_metric="train_loss",
            #     )
            #     opter_config_for_run["lr"] = best_lr
            #     print(f"  Best lr for {opter_names} optimizers: {best_lr}")

            ### dump config
            run_config["meta_testing"]["baseline_opter_names"] = opter_names
            run_config["meta_testing"]["baseline_opter_cls_list"] = opter_cls_list
            run_config["meta_testing"]["baseline_opter_configs"] = opter_configs_for_run
            run_config["meta_testing"]["baseline_opter_mul_factors"] = opter_mul_factors
            if run_config["meta_testing"]["ckpt_iter_freq"] is not None:
                with open(os.path.join(baseline_opter_dir, "config.json"), "w") as f:
                    json.dump(run_config, f, indent=4, default=str)
                torch.save(run_config, os.path.join(baseline_opter_dir, "config.pt"))

            ### train
            baseline_ckpt_dir = os.path.join(baseline_opter_dir, "ckpt") if run_config["meta_testing"]["ckpt_iter_freq"] is not None else None
            baseline_metrics = fit_normal(
                data_cls=run_config["meta_testing"]["data_cls"],
                data_config=run_config["meta_testing"]["data_config"],
                optee_cls=run_config["meta_testing"]["optee_cls"],
                optee_config=run_config["meta_testing"]["optee_config"],
                opter_cls_list=opter_cls_list,
                opter_configs=opter_configs_for_run,
                opter_mul_factors=opter_mul_factors,
                n_iters=run_config["meta_testing"]["n_iters"],
                n_tests=run_config["eval_n_tests"],
                ckpt_iter_freq=run_config["meta_testing"]["ckpt_iter_freq"],
                ckpt_dir=baseline_ckpt_dir,
                save_ckpts_for_all_test_runs=save_ckpts_for_all_test_runs,
            )

            ### save all info to disk as .pt (config and metrics)
            run_info = {
                "config": run_config,
                "metrics": baseline_metrics,
            }
            results[f"{'-'.join(opter_names)}_{optee_cls.__name__}_{dict_to_str(optee_config)}"] = run_info

            ### save metrics to disk
            if run_config["meta_testing"]["ckpt_iter_freq"] is not None:
                np.save(metrics_path, baseline_metrics)
                print(f"  Metrics of {opter_names} saved to {metrics_path}")

                run_info_path = os.path.join(baseline_opter_dir, "run.pt")
                torch.save(run_info, run_info_path)

    return results


def get_ckpt_dir(
    opter_cls_list,
    opter_configs,
    opter_mul_factors,
    optee_cls,
    optee_config,
    data_cls,
    data_config,
):
    ckpt_dir = "-".join(opter_cls.__name__ for opter_cls in opter_cls_list)
    ckpt_dir += "_"
    ckpt_dir += "-".join(dict_to_str(opter_config) for opter_config in opter_configs)
    ckpt_dir += "_"
    ckpt_dir += "-".join(str(opter_mul_factor) for opter_mul_factor in opter_mul_factors)
    ckpt_dir += "_"
    ckpt_dir += optee_cls.__name__
    ckpt_dir += "_"
    ckpt_dir += dict_to_str(optee_config)
    ckpt_dir += "_"
    ckpt_dir += data_cls.__name__
    ckpt_dir += "_"
    ckpt_dir += dict_to_str(data_config)
    return ckpt_dir
