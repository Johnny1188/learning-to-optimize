import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm.notebook import tqdm

from meta_module import *
from optimizer import Optimizer
from utils.others import count_parameters, detach_var, rgetattr, rsetattr, w


def do_fit(
    opter,
    opter_optim,
    target_cls,
    optee_cls,
    unroll,
    n_iters,
    optee_updates_lr,
    train_opter=True,
    optee_config=None,
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

    target = target_cls(training=train_opter)
    optee = w(optee_cls(**optee_config if optee_config is not None else {}))
    optee_n_params = sum(
        [int(np.prod(p.size())) for _, p in optee.all_named_parameters()]
    )

    ### initialize hidden and cell states
    hidden_states = [
        w(Variable(torch.zeros(optee_n_params, opter.hidden_sz))) for _ in range(2)
    ]
    cell_states = [
        w(Variable(torch.zeros(optee_n_params, opter.hidden_sz))) for _ in range(2)
    ]

    all_losses = []
    unroll_losses = None
    updates_for_ckpt = dict()
    for iteration in range(1, n_iters + 1):
        loss = optee(target)

        unroll_losses = loss if unroll_losses is None else unroll_losses + loss
        all_losses.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=train_opter)

        ### optimizer: gradients -> updates
        result_params = {}
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
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1))
            updates, new_hidden, new_cell = opter(
                gradients,
                [h[offset : offset + cur_sz] for h in hidden_states],
                [c[offset : offset + cur_sz] for c in cell_states],
            )
            if ckpt_iter_freq and iteration % ckpt_iter_freq == 0:
                updates_for_ckpt[name] = updates.view(*p.size()).detach()
            for i in range(len(new_hidden)):
                hidden_states2[i][offset : offset + cur_sz] = new_hidden[i]
                cell_states2[i][offset : offset + cur_sz] = new_cell[i]
            result_params[name] = p + optee_updates_lr * updates.view(*p.size())
            result_params[name].retain_grad()

            offset += cur_sz

        ### checkpoint
        if ckpt_iter_freq and iteration % ckpt_iter_freq == 0:
            ckpt = {
                "optimizee": optee.state_dict(),
                "optimizee_grads": {k: v.grad for k, v in optee.all_named_parameters()},
                "optimizee_updates": updates_for_ckpt,
                "optimizer": opter.state_dict(),
                "loss_history": all_losses,
            }
            torch.save(ckpt, os.path.join(ckpt_dir, f"{ckpt_prefix}{iteration}.pt"))
            updates_for_ckpt = dict()

        ### update - continue unrolling or step w/ opter
        if iteration % unroll == 0:
            # step w/ the optimizer
            if train_opter:
                opter_optim.zero_grad()
                unroll_losses.backward()
                opter_optim.step()

            # reinitialize - start next unroll
            optee = w(optee_cls(**optee_config if optee_config is not None else {}))
            optee.load_state_dict(result_params)
            optee.zero_grad()
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]
            unroll_losses = None
        else:
            # update the optimizee and optimizer's states
            for name, p in optee.all_named_parameters():
                if p.requires_grad:  # batchnorm stats
                    rsetattr(optee, name, result_params[name])
            hidden_states = hidden_states2
            cell_states = cell_states2

    return all_losses


def fit_optimizer(
    target_cls,
    optee_cls,
    optee_config=None,
    preproc=False,
    unroll=20,
    n_epochs=20,
    n_optim_runs_per_epoch=20,
    n_iters=100,
    n_tests=100,
    opter_lr=0.001,
    optee_updates_lr=1.0,
    ckpt_iter_freq=None,
    ckpt_prefix="",
    ckpt_dir="",
):
    if ckpt_iter_freq is not None:
        os.makedirs(ckpt_dir, exist_ok=True)

    opt_net = w(Optimizer(preproc=preproc))
    meta_opt = optim.Adam(opt_net.parameters(), lr=opter_lr)

    best_net = None
    best_loss = np.inf

    for epoch_i in tqdm(range(n_epochs), "epochs"):
        ### train
        train_loss = 0.0
        for _ in tqdm(range(n_optim_runs_per_epoch), "optimization runs"):
            train_loss += np.sum(
                do_fit(
                    opter=opt_net,
                    opter_optim=meta_opt,
                    target_cls=target_cls,
                    optee_cls=optee_cls,
                    optee_config=optee_config,
                    unroll=unroll,
                    n_iters=n_iters,
                    optee_updates_lr=optee_updates_lr,
                    train_opter=True,
                    ckpt_iter_freq=ckpt_iter_freq,
                    ckpt_prefix=f"{ckpt_prefix}{epoch_i}e_",
                    ckpt_dir=ckpt_dir,
                )
            )
        train_loss /= n_optim_runs_per_epoch
        print(f"[{epoch_i + 1}/{n_epochs}] Training loss: {train_loss}")

        ### test
        test_loss = 0.0
        for _ in tqdm(range(n_tests), "tests"):
            test_loss += np.sum(
                do_fit(
                    opter=opt_net,
                    opter_optim=meta_opt,
                    target_cls=target_cls,
                    optee_cls=optee_cls,
                    optee_config=optee_config,
                    unroll=unroll,
                    n_iters=n_iters,
                    optee_updates_lr=optee_updates_lr,
                    train_opter=False,
                    ckpt_iter_freq=None,
                )
            )
        test_loss /= n_tests
        print(f"[{epoch_i + 1}/{n_epochs}] Testing loss: {test_loss}")

        if test_loss < best_loss:
            print(
                f"[{epoch_i + 1}/{n_epochs}] New best loss\n\t previous: \t{best_loss}\n\t current:\t {test_loss}"
            )
            best_loss = test_loss
            best_net = copy.deepcopy(opt_net.state_dict())

    return best_loss, best_net


def fit_normal(
    target_cls,
    optee_cls,
    opter_cls,
    n_tests=100,
    n_iters=100,
    optee_config=None,
    ckpt_iter_freq=None,
    ckpt_prefix="",
    ckpt_dir="",
    **kwargs,
):
    if ckpt_iter_freq is not None:
        os.makedirs(ckpt_dir, exist_ok=True)

    results = []
    for _ in tqdm(range(n_tests), f"{opter_cls.__name__} - tests"):
        target = target_cls(training=False)
        optee = w(optee_cls(**optee_config if optee_config is not None else {}))
        opter = opter_cls(optee.parameters(), **kwargs)
        loss_history = []
        for iter_i in range(1, n_iters + 1):
            loss = optee(target)
            opter.zero_grad()
            loss.backward()

            ### model checkpointing
            if ckpt_iter_freq and iter_i % ckpt_iter_freq == 0:
                ckpt = {
                    "optimizee": optee.state_dict(),
                    "optimizee_grads": {
                        k: v.grad for k, v in optee.all_named_parameters()
                    },
                    "optimizer": opter.state_dict(),
                    "loss_history": loss_history,
                }
                torch.save(ckpt, os.path.join(ckpt_dir, f"{ckpt_prefix}{iter_i}.pt"))

            ### update (after checkpointing)
            opter.step()
            loss_history.append(loss.data.cpu().numpy())
        results.append(loss_history)
    return results


# def find_best_lr_normal(target_cls, optee_cls, opter_cls, **extra_kwargs):
#     best_loss = np.inf
#     best_lr = 0.0
#     for lr in tqdm([1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001], 'Learning rates'):
#         try:
#             loss = best_loss + 1.0
#             loss = np.mean([np.sum(s) for s in fit_normal(target_cls=target_cls,
#                 optee_cls=optee_cls, opter_cls=opter_cls, lr=lr, **extra_kwargs)])
#         except RuntimeError:
#             pass
#         if loss < best_loss:
#             best_loss = loss
#             best_lr = lr
#     return best_loss, best_lr
