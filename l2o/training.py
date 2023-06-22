import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from l2o.meta_module import *
from l2o.optimizer import Optimizer
from l2o.others import count_parameters, detach_var, rgetattr, rsetattr, w


def do_fit(
    opter,
    opter_optim,
    data_cls,
    optee_cls,
    unroll,
    n_iters,
    optee_updates_lr,
    data_config=None,
    train_opter=True,
    log_unroll_losses=False,
    opter_updates_reg_func=None,
    opter_updates_reg_func_config=None,
    reg_mul=1.0,
    optee_config=None,
    eval_iter_freq=10,
    ckpt_iter_freq=None,
    ckpt_prefix="",
    ckpt_dir="",
):
    if train_opter:
        opter.train()
        opter_optim.zero_grad()
    else:
        opter.eval()
        unroll = 1

    train_data = data_cls(
        training=True, **data_config if data_config is not None else {}
    )
    test_data = data_cls(
        training=False, **data_config if data_config is not None else {}
    )
    optee = w(optee_cls(**optee_config if optee_config is not None else {}))
    optee_n_params = sum(
        [int(np.prod(p.size())) for _, p in optee.all_named_parameters() if p.requires_grad]
    )

    ### save initial optee's parameters for regularization
    optee_init_params = dict()
    for name, p in optee.all_named_parameters():
        optee_init_params[name] = p.data.detach().clone()

    ### initialize hidden and cell states
    hidden_states = [
        w(Variable(torch.zeros(optee_n_params, opter.hidden_sz))) for _ in range(2)
    ]
    cell_states = [
        w(Variable(torch.zeros(optee_n_params, opter.hidden_sz))) for _ in range(2)
    ]

    metrics = {m: [] for m in ["train_loss", "train_acc", "test_loss", "test_acc"]}
    unroll_losses = None
    reg_losses = None
    updates_for_ckpt = dict()

    ### run optee's training loop
    for iteration in range(1, n_iters + 1):
        ### train optee
        optee.train()
        train_loss, train_acc = optee(train_data, return_acc=True)  # a single minibatch
        train_loss.backward(retain_graph=train_opter)

        ### track for training opter
        unroll_losses = (
            train_loss if unroll_losses is None else unroll_losses + train_loss
        )

        ### optimizer: gradients -> updates
        result_params = dict()
        updates_for_reg = dict()
        hidden_states2 = [
            w(Variable(torch.zeros(optee_n_params, opter.hidden_sz))) for _ in range(2)
        ]
        cell_states2 = [
            w(Variable(torch.zeros(optee_n_params, opter.hidden_sz))) for _ in range(2)
        ]
        offset = 0
        for name, p in optee.all_named_parameters():
            if p.requires_grad == False:  # batchnorm stats
                result_params[name] = p
                continue

            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            cur_sz = int(np.prod(p.size()))
            gradients = detach_var(p.grad.view(cur_sz, 1))
            updates, new_hidden, new_cell = opter(
                optee_grads=gradients,
                hidden=[h[offset : offset + cur_sz] for h in hidden_states],
                cell=[c[offset : offset + cur_sz] for c in cell_states],
                additional_inp=None,
            )

            ### track updates for checkpointing
            if ckpt_iter_freq and (iteration % ckpt_iter_freq == 0 or iteration == 1):
                updates_for_ckpt[name] = updates.view(*p.size()).detach()

            ### track updates for regularization
            if train_opter and opter_updates_reg_func is not None:
                updates_for_reg[name] = updates.view(*p.size())

            ### update hidden and cell states
            for i in range(len(new_hidden)):
                hidden_states2[i][offset : offset + cur_sz] = new_hidden[i]
                cell_states2[i][offset : offset + cur_sz] = new_cell[i]

            ### update optee's params
            result_params[name] = p + optee_updates_lr * updates.view(*p.size())
            result_params[name].retain_grad()
            offset += cur_sz

        ### add regularization loss for opter
        if train_opter and opter_updates_reg_func is not None:
            if "conservation_law_breaking" in opter_updates_reg_func.__name__:  # TODO: quick hack
                reg_loss = torch.abs(
                    reg_mul
                    * opter_updates_reg_func(
                        optee=optee,
                        params_t0=optee_init_params,
                        **opter_updates_reg_func_config
                        if opter_updates_reg_func_config is not None
                        else {},
                    )
                )
            else:
                reg_loss = torch.abs(
                    reg_mul
                    * opter_updates_reg_func(
                        updates=updates_for_reg,
                        optee=optee,
                        lr=optee_updates_lr,
                        **opter_updates_reg_func_config
                        if opter_updates_reg_func_config is not None
                        else {},
                    )
                )
            reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss
            updates_for_reg = dict()
            # add to metrics
            if "train_reg_loss" not in metrics:
                metrics["train_reg_loss"] = []
            metrics["train_reg_loss"].append(reg_loss.item())

        ### track metrics
        metrics["train_loss"].append(train_loss.item())
        metrics["train_acc"].append(train_acc.item())

        ### eval
        if eval_iter_freq is not None and (iteration % eval_iter_freq == 0 or iteration == 1):
            optee.eval()
            test_loss, test_acc = optee(test_data, return_acc=True)
            metrics["test_loss"].append(test_loss.item())
            metrics["test_acc"].append(test_acc.item())
            optee.train()

        ### checkpoint
        if ckpt_iter_freq and (iteration % ckpt_iter_freq == 0 or iteration == 1):
            ckpt = {
                "optimizee": optee.state_dict(),
                "optimizee_grads": {k: v.grad for k, v in optee.all_named_parameters()},
                "optimizee_updates": updates_for_ckpt,
                "optimizer": opter.state_dict(),
                "hidden_states": hidden_states,
                "cell_states": cell_states,
                "metrics": metrics,
            }
            if not ckpt_dir.startswith(os.environ["CKPT_PATH"]):
                print(f"[WARNING] ckpt_dir {ckpt_dir} does not start with CKPT_PATH, prepending CKPT_PATH to it")
                ckpt_dir = os.path.join(os.environ["CKPT_PATH"], ckpt_dir)
            torch.save(ckpt, os.path.join(ckpt_dir, f"{ckpt_prefix}{iteration}.pt"))
            updates_for_ckpt = dict()

        ### update - continue unrolling or step w/ opter
        if iteration % unroll == 0:
            ### step w/ the optimizer
            if train_opter:
                opter_optim.zero_grad()
                if log_unroll_losses:
                    unroll_losses = torch.log(unroll_losses)
                total_loss = (
                    unroll_losses + reg_losses
                    if reg_losses is not None
                    else unroll_losses
                )
                total_loss.backward()
                opter_optim.step()

            ### reinitialize - start next unroll
            optee = w(optee_cls(**optee_config if optee_config is not None else {}))
            optee.load_state_dict(result_params)
            optee.zero_grad()
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]
            unroll_losses = None
            reg_losses = None
        else:
            ### update the optimizee and optimizer's states
            for name, p in optee.all_named_parameters():
                if p.requires_grad:  # leave the batchnorm stats
                    rsetattr(optee, name, result_params[name])
            hidden_states = hidden_states2
            cell_states = cell_states2

    return metrics


def fit_optimizer(
    data_cls,
    optee_cls,
    data_config=None,
    optee_config=None,
    opter_cls=Optimizer,
    opter_config=None,
    unroll=20,
    n_epochs=20,
    n_optim_runs_per_epoch=20,
    n_iters=100,
    n_tests=100,
    opter_lr=0.01,
    log_unroll_losses=False,
    opter_updates_reg_func=None,
    opter_updates_reg_func_config=None,
    reg_mul=1.0,
    optee_updates_lr=1.0,
    eval_iter_freq=10,
    ckpt_iter_freq=None,
    ckpt_epoch_freq=None,
    ckpt_prefix="",
    ckpt_dir="",
    load_ckpt=None,
    start_from_epoch=0,
    verbose=1,
):
    if ckpt_iter_freq is not None:
        os.makedirs(ckpt_dir, exist_ok=True)

    opter = w(opter_cls(**opter_config if opter_config is not None else {}))
    meta_opt = optim.Adam(opter.parameters(), lr=opter_lr)

    best_opter = None
    best_loss = np.inf
    all_metrics = list()
    
    ### load checkpoint
    if load_ckpt is not None:
        print(f"... loading checkpoint from {load_ckpt} ...")
        ckpt = torch.load(load_ckpt)
        best_opter = ckpt["best_opter"]
        best_loss = ckpt["best_loss"]
        all_metrics = ckpt["metrics"]
        opter.load_state_dict(ckpt["opter"])
        meta_opt = optim.Adam(opter.parameters(), lr=opter_lr)
        meta_opt.load_state_dict(ckpt["meta_opter"])
        meta_opt.zero_grad()
        opter.train()

    ### meta-training epochs
    for epoch_i in range(start_from_epoch, n_epochs):
        start_time = time.time()
        if verbose > 0:
            print(f"[{epoch_i + 1}/{n_epochs}]")
        all_metrics.append({k: dict() for k in ["meta_training", "meta_testing"]})

        ### meta-train
        for run_i in range(n_optim_runs_per_epoch):
            curr_ckpt_iter_freq = None
            if epoch_i % ckpt_epoch_freq == 0 and run_i == n_optim_runs_per_epoch - 1:
                curr_ckpt_iter_freq = ckpt_iter_freq
            optim_run_metrics = do_fit(
                opter=opter,
                opter_optim=meta_opt,
                data_cls=data_cls,
                data_config=data_config,
                optee_cls=optee_cls,
                optee_config=optee_config,
                unroll=unroll,
                n_iters=n_iters,
                optee_updates_lr=optee_updates_lr,
                train_opter=True,
                log_unroll_losses=log_unroll_losses,
                opter_updates_reg_func=opter_updates_reg_func,
                opter_updates_reg_func_config=opter_updates_reg_func_config,
                reg_mul=reg_mul,
                eval_iter_freq=eval_iter_freq,
                ckpt_iter_freq=curr_ckpt_iter_freq,
                ckpt_prefix=f"{ckpt_prefix}{epoch_i}e_",
                ckpt_dir=ckpt_dir,
            )
            log_msg = f"  [{run_i + 1}/{n_optim_runs_per_epoch}]"
            for k, v in optim_run_metrics.items():
                if k not in all_metrics[-1]["meta_training"]:
                    all_metrics[-1]["meta_training"][k] = np.array(v)
                else:
                    all_metrics[-1]["meta_training"][k] += np.array(v)
                if "acc" in k:
                    log_msg += f"  {k}_last={v[-1]:.3f}"
                else:
                    log_msg += f"  {k}_sum={np.sum(v):.3f}  {k}_last={v[-1]:.3f}"
            if verbose > 1:
                print(log_msg)

        ### average metrics and log
        for k, v in all_metrics[-1]["meta_training"].items():
            v = v / n_optim_runs_per_epoch
            all_metrics[-1]["meta_training"][k] = {"sum": np.sum(v), "last": v[-1]}

        if verbose > 0:
            print(
                f"[{epoch_i + 1}/{n_epochs}] Meta-training metrics:"
                f"\n{json.dumps(all_metrics[-1]['meta_training'], indent=4, sort_keys=False)}"
            )

        ### meta-test
        if n_tests > 0:
            for _ in range(n_tests):
                optim_run_metrics = do_fit(
                    opter=opter,
                    opter_optim=meta_opt,
                    data_cls=data_cls,
                    data_config=data_config,
                    optee_cls=optee_cls,
                    optee_config=optee_config,
                    unroll=unroll,
                    n_iters=n_iters,
                    optee_updates_lr=optee_updates_lr,
                    train_opter=False,
                    eval_iter_freq=eval_iter_freq,
                    ckpt_iter_freq=None,
                )
                for k, v in optim_run_metrics.items():
                    if k not in all_metrics[-1]["meta_testing"]:
                        all_metrics[-1]["meta_testing"][k] = np.array(v)
                    else:
                        all_metrics[-1]["meta_testing"][k] += np.array(v)

            ### average metrics and log
            for k, v in all_metrics[-1]["meta_testing"].items():
                v = v / n_tests
                all_metrics[-1]["meta_testing"][k] = {"sum": np.sum(v), "last": v[-1]}

            if verbose > 0:
                print(
                    f"[{epoch_i + 1}/{n_epochs}] Meta-testing metrics:"
                    f"\n{json.dumps(all_metrics[-1]['meta_testing'], indent=4, sort_keys=False)}"
                )

            if all_metrics[-1]["meta_testing"]["train_loss"]["sum"] < best_loss:
                if verbose > 0:
                    print(
                        f"[{epoch_i + 1}/{n_epochs}] New best loss"
                        f"\n\t previous:\t {best_loss}"
                        f"\n\t current:\t {all_metrics[-1]['meta_testing']['train_loss']['sum']} (at last iter: {all_metrics[-1]['meta_testing']['train_loss']['last']})"
                    )
                best_loss = all_metrics[-1]["meta_testing"]["train_loss"]["sum"]
                best_opter = copy.deepcopy(opter.state_dict())
        else:
            ### no meta-testing, so just save the best model based on meta-training
            if all_metrics[-1]["meta_training"]["train_loss"]["sum"] < best_loss:
                if verbose > 0:
                    print(
                        f"[{epoch_i + 1}/{n_epochs}] New best loss"
                        f"\n\t previous:\t {best_loss}"
                        f"\n\t current:\t {all_metrics[-1]['meta_training']['train_loss']['sum']:.3f} (at last iter: {all_metrics[-1]['meta_training']['train_loss']['last']:.3f})"
                    )
                best_loss = all_metrics[-1]["meta_training"]["train_loss"]["sum"]
                best_opter = copy.deepcopy(opter.state_dict())

        ### save ckpt
        if epoch_i % ckpt_epoch_freq == 0:
            ckpt = {
                "best_opter": best_opter if best_opter else None,
                "best_loss": best_loss,
                "opter": opter.state_dict(),
                "meta_opter": meta_opt.state_dict(),
                "metrics": all_metrics,
            }
            torch.save(ckpt, os.path.join(ckpt_dir, f"{epoch_i}.pt"))

        end_time = time.time()
        if verbose > 0:
            print(f"[{epoch_i + 1}/{n_epochs}] Epoch took {end_time - start_time:.2f}s")

    return best_loss, all_metrics, best_opter


def fit_normal(
    data_cls,
    optee_cls,
    opter_cls,
    n_tests=100,
    n_iters=100,
    data_config=None,
    optee_config=None,
    opter_config=None,
    eval_iter_freq=10,
    ckpt_iter_freq=None,
    ckpt_dir="",
    save_ckpts_for_all_test_runs=False,
):
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
        opter = opter_cls(optee.parameters(), **opter_config)

        ### new test run
        for k in metrics.keys():
            metrics[k].append([])

        for iter_i in range(1, n_iters + 1):
            ### train
            optee.train()
            opter.zero_grad()
            train_loss, train_acc = optee(train_data, return_acc=True)
            train_loss.backward()

            ### model checkpointing - save only for the first test run if save_ckpts_for_all_test_runs is False
            if ckpt_iter_freq and (iter_i % ckpt_iter_freq == 0 or iter_i == 1) \
                and (test_i == 0 or save_ckpts_for_all_test_runs):
                ckpt = {
                    "optimizee": optee.state_dict(),
                    "optimizee_grads": {
                        k: v.grad for k, v in optee.all_named_parameters()
                    },
                    "optimizer": opter.state_dict(),
                    "metrics": metrics,
                }
                torch.save(ckpt, os.path.join(ckpt_dir, f"run{test_i}_{iter_i}.pt"))

            ### update (after checkpointing)
            opter.step()

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
