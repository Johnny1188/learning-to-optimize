import json
import os
from copy import deepcopy

import numpy as np
import torch

from l2o.others import dict_to_str, get_baseline_ckpt_dir
from l2o.training import do_fit, fit_normal


def meta_test(opter, optees, config, save_ckpts_for_all_test_runs=False, seed=0):
    """
    Parameters
    ----------
    opter : l2o.optimizer.Optimizer
        The L2O optimizer model to meta-test.
    optees : list of tuples
        List of tuples of the form (optimizee_cls, optimizee_config) to meta-test on.
    config : dict
        The config dictionary for the meta-testing run.
    save_ckpts_for_all_test_runs : bool, optional
        Whether to save checkpoints for all test runs, by default False.
    seed : int, optional
        The random seed to use for the meta-testing run, by default 0.

    Returns
    -------
    results : dict
        Dictionary of results.
    """
    ### meta-test the L2O optimizer model on various optimizees
    results = dict()
    for optee_cls, optee_config in optees:
        meta_task_name = (
            f"{optee_cls.__name__}_{dict_to_str(optee_config)}"
            + f"_{config['meta_testing']['data_cls'].__name__}_{dict_to_str(config['meta_testing']['data_config'])}"
        )
        print(f"Meta-testing on {meta_task_name}")

        ### set config
        run_config = deepcopy(config)
        run_config["meta_testing"]["optee_cls"] = optee_cls
        run_config["meta_testing"]["optee_config"] = optee_config
        if save_ckpts_for_all_test_runs is False:
            ckpt_prefix = ""

        ### run the meta-testing run with l2o optimizer
        torch.manual_seed(seed)
        np.random.seed(seed)
        metrics = []
        for eval_test_i in range(run_config["eval_n_tests"]):
            ckpt_prefix = f"run{eval_test_i}_" # add prefix to checkpoints
            if not save_ckpts_for_all_test_runs and eval_test_i > 0:
                run_config["meta_testing"]["ckpt_iter_freq"] = None # only save checkpoints for the first run
            metrics.append(
                do_fit(
                    opter=opter,
                    **run_config["meta_testing"],
                    ckpt_prefix=ckpt_prefix,
                )
            )

        ### aggregate metrics by metric name and save
        metrics = {k: np.array([m[k] for m in metrics]) for k in metrics[0].keys()}
        metrics_path = os.path.join(
            os.environ["CKPT_PATH"], run_config["ckpt_base_dir"], f"metrics_{meta_task_name}.npy"
        )
        np.save(metrics_path, metrics)
        print(f"  Metrics saved to {metrics_path}")

        ### save the config for this run
        config_path = os.path.join(
            os.environ["CKPT_PATH"], run_config["ckpt_base_dir"], f"config_{meta_task_name}.json"
        )
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=4, default=str)
        
        ### save also the whole run information as .pt (config and metrics)
        run_info = {
            "config": run_config,
            "metrics": metrics,
        }
        run_info_path = os.path.join(
            os.environ["CKPT_PATH"], run_config["ckpt_base_dir"], f"run_{meta_task_name}.pt"
        )
        torch.save(run_info, run_info_path)
        
        results[meta_task_name] = run_info

    return results


def meta_test_baselines(baseline_opters, optees, config, use_existing_baselines=True, save_ckpts_for_all_test_runs=False):
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
    results = dict()
    for optee_cls, optee_config in optees:
        ### train optees with baseline optimizers (or load previous)
        for opter_name, baseline_opter_cls, opter_config in baseline_opters:
            print(f"Training {optee_cls.__name__}_{dict_to_str(optee_config)} with {opter_name} optimizer")
            torch.manual_seed(0)
            np.random.seed(0)

            ### set config
            run_config = deepcopy(config)
            run_config["meta_testing"]["optee_cls"] = optee_cls
            run_config["meta_testing"]["optee_config"] = optee_config

            ### prepare checkpointing
            baseline_file_nickname = get_baseline_ckpt_dir(
                opter_cls=baseline_opter_cls,
                opter_config=opter_config,
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

            ### load previous if exists
            if use_existing_baselines and os.path.exists(metrics_path) and os.path.isdir(os.path.join(baseline_opter_dir, "ckpt")):
                print(
                    f"  Existing metrics and checkpoints for {opter_name} exist, skipping..."
                    f"\n  metrics_file: {metrics_path}"
                )
                continue

            ### not reusing existing, train from scratch
            opter_config_for_run = deepcopy(opter_config)

            ### find best lr
            if "lr" in opter_config_for_run and callable(opter_config_for_run["lr"]):
                print(f"  Finding best lr for {opter_name} optimizer")
                best_lr = opter_config_for_run["lr"](
                    data_cls=run_config["meta_testing"]["data_cls"],
                    optee_cls=run_config["meta_testing"]["optee_cls"],
                    opter_cls=baseline_opter_cls,
                    n_tests=3,
                    n_iters=run_config["meta_testing"]["n_iters"] // 2,
                    optee_config=run_config["meta_testing"]["optee_config"],
                    opter_config=opter_config_for_run,
                    consider_metric="train_loss",
                )
                opter_config_for_run["lr"] = best_lr
                print(f"  Best lr for {opter_name} optimizer: {best_lr}")

            ### dump config
            run_config["meta_testing"]["baseline_opter_config"] = opter_config_for_run
            with open(os.path.join(baseline_opter_dir, "config.json"), "w") as f:
                json.dump(run_config, f, indent=4, default=str)
            torch.save(run_config, os.path.join(baseline_opter_dir, "config.pt"))

            ### train
            baseline_ckpt_dir = os.path.join(baseline_opter_dir, "ckpt")
            baseline_metrics = fit_normal(
                data_cls=run_config["meta_testing"]["data_cls"],
                data_config=run_config["meta_testing"]["data_config"],
                optee_cls=run_config["meta_testing"]["optee_cls"],
                optee_config=run_config["meta_testing"]["optee_config"],
                opter_cls=baseline_opter_cls,
                opter_config=opter_config_for_run,
                n_iters=run_config["meta_testing"]["n_iters"],
                n_tests=run_config["eval_n_tests"],
                ckpt_iter_freq=run_config["meta_testing"]["ckpt_iter_freq"],
                ckpt_dir=baseline_ckpt_dir,
                save_ckpts_for_all_test_runs=save_ckpts_for_all_test_runs,
            )

            ### save metrics to disk
            np.save(metrics_path, baseline_metrics)
            print(f"  Metrics of {opter_name} saved to {metrics_path}")

            ### save all info to disk as .pt (config and metrics)
            run_info = {
                "config": run_config,
                "metrics": baseline_metrics,
            }
            run_info_path = os.path.join(baseline_opter_dir, "run.pt")
            torch.save(run_info, run_info_path)

            results[f"{opter_name}_{optee_cls.__name__}_{dict_to_str(optee_config)}"] = run_info

    return results
