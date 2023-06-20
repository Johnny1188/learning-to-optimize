import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from l2o.others import (load_baseline_opter_ckpt, load_ckpt,
                        load_l2o_opter_ckpt, w)


def get_rescale_sym_constraint_deviation(
    W1,
    b,
    W2,
    W1_update,
    b_update,
    W2_update,
):
    """Applies to ReLU, LeakyReLU, Linear, ...
    inner(W1_gradient, W1) + inner(b_gradient, b) - inner(W2_gradient, W2) = 0

    See rescale symmetry in section 3 of the Neural Mechanics paper
    (https://arxiv.org/abs/2012.04728)

    - Note: The left hand side of the constraint should be zero for gradients,
    which are negative updates in the case of SGD (just scaled by lr), hence
    any other parameter updates (from Adam, L2O optimizer, etc.) need to be
    negated (otherwise the deviations will have the opposite sign).
    """
    # if normalize_updates:
    #     W1_update = F.normalize(W1_update, dim=0)
    #     b_update = b_update / (b_update.norm() + 1e-8)
    #     W2_update = F.normalize(W2_update, dim=0)

    # if normalize_params:
    #     W1 = W1 / (W1.norm() + 1e-8)
    #     b = b / (b.norm() + 1e-8)
    #     W2 = W2 / (W2.norm() + 1e-8)

    return (
        W1_update.flatten() @ W1.flatten()
        + b_update.flatten() @ b.flatten()
        - W2_update.flatten() @ W2.flatten()
    )


def get_translation_sym_constraint_deviations(
    W_update, b_update
):
    """Applies to Softmax
    inner(W_gradient, all_ones) = inner(b_gradient, all_ones) = 0

    See translation symmetry in section 3 of the Neural Mechanics paper
    (https://arxiv.org/abs/2012.04728)

    - Note: The left hand side of the constraint should be zero for gradients,
    which are negative updates in the case of SGD (just scaled by lr), hence
    any other parameter updates (from Adam, L2O optimizer, etc.) need to be
    negated (otherwise the deviations will have the opposite sign).
    - Note: The way inner_W_update is computed is a shortcut for the following:
        W_constraint_deviation = 0
        all_ones = torch.ones(W_update.shape[0], 1)
        for preceding_layer_neuron_i in range(W_update.shape[1]):
            W_constraint_deviation += (W_update[:, preceding_layer_neuron_i] @ all_ones).item()
    """
    # if normalize_updates:
    #     W_update = F.normalize(W_update, dim=0)
    #     b_update = b_update / (b_update.norm() + 1e-8)

    W_deviations = W_update.sum()
    b_deviations = b_update.sum()  # inner(b_update, all_ones)

    return W_deviations, b_deviations


def get_scale_sym_constraint_deviation(
    W, b, W_update, b_update
):
    """Applies to Batch normalization
    inner(W_gradient, W) + inner(b_gradient, b) = 0

    See scale symmetry in section 3 of the Neural Mechanics paper
    (https://arxiv.org/abs/2012.04728)

    - Note: The left hand side of the constraint should be zero for gradients,
    which are negative updates in the case of SGD (just scaled by lr), hence
    any other parameter updates (from Adam, L2O optimizer, etc.) need to be
    negated (otherwise the deviations will have the opposite sign).
    """
    # if normalize_updates:
    #     W_update = F.normalize(W_update, dim=0)
    #     b_update = b_update / (b_update.norm() + 1e-8)

    # if normalize_params:
    #     W = W / (W.norm() + 1e-8)
    #     b = b / (b.norm() + 1e-8)

    W_deviations = W_update.flatten() @ W.flatten()
    b_deviations = b_update.flatten() @ b.flatten()
    return W_deviations + b_deviations


def get_adam_param_update(
    param, grad, opter_state, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
):
    beta1, beta2 = betas

    if weight_decay != 0.0:
        grad = grad + weight_decay * param

    biased_first_moment = beta1 * opter_state["exp_avg"] + (1 - beta1) * grad
    biased_second_moment = beta2 * opter_state["exp_avg_sq"] + (1 - beta2) * (grad**2)

    # bias correction
    bias_correction1 = 1 - beta1 ** (opter_state["step"] + 1)
    bias_correction2 = 1 - beta2 ** (opter_state["step"] + 1)
    bias_corrected_first_moment = biased_first_moment / bias_correction1
    bias_corrected_second_moment = biased_second_moment / bias_correction2

    # update (new_param = old_param + param_update)
    param_update = (
        -lr * bias_corrected_first_moment / (torch.sqrt(bias_corrected_second_moment) + eps)
    )

    return param_update


def get_baseline_opter_param_updates(optee, opter, verbose=True):
    optee_updates = {}
    opter_state_dict = opter.state_dict()
    if opter_state_dict["state"] is None or len(opter_state_dict["state"]) == 0: # first step
        if verbose:
            print(f"[WARNING] opter_state_dict['state'] is None, initializing")
        opter_state_dict["state"] = [{
            "exp_avg": torch.zeros_like(p),
            "exp_avg_sq": torch.zeros_like(p),
            "step": 0,
        } for _, p in optee.all_named_parameters()]

    if isinstance(opter, optim.Adam):
        for p_i, (n, p) in enumerate(optee.all_named_parameters()):
            if p.grad is None:
                if verbose:
                    print(f"[WARNING] p.grad is None for {n}, skipping")
                continue
            assert p.shape == p.grad.shape
            assert p.shape == opter_state_dict["state"][p_i]["exp_avg"].shape
            assert p.shape == opter_state_dict["state"][p_i]["exp_avg_sq"].shape
            optee_updates[n] = get_adam_param_update(
                param=p,
                grad=p.grad,
                opter_state=opter_state_dict["state"][p_i],
                lr=opter_state_dict["param_groups"][0]["lr"],
                betas=opter_state_dict["param_groups"][0]["betas"],
                eps=opter_state_dict["param_groups"][0]["eps"],
                weight_decay=opter_state_dict["param_groups"][0]["weight_decay"],
            )
    elif isinstance(opter, optim.SGD):
        for n, p in optee.all_named_parameters():
            if p.grad is not None:
                optee_updates[n] = -opter_state_dict["param_groups"][0]["lr"] * p.grad
    else:
        raise NotImplementedError(f"Optimizer {type(opter)} not implemented")
    return optee_updates


def validate_inputs_for_collecting_deviations(opter_name, phase):
    assert opter_name in [
        "Optimizer",
        "SGD",
        "Adam",
    ], f"opter_cls {opter_name} not supported"
    assert phase in ["meta_training", "meta_testing"], f"phase {phase} not supported"
    assert (
        phase != "meta_training" or opter_name == "Optimizer"
    ), f"opter_cls {opter_name} not supported for phase {phase}"
    return True


def collect_rescale_sym_deviations(
    ckpt_iter_freq,
    n_iters,
    optee_cls,
    opter_cls,
    optee_config=None,
    opter_config=None,
    phase="meta_testing",
    ckpt_path_prefix="",
    max_iters=None,
):
    """
    Collects get_rescale_sym_constraint_deviation() for all checkpoints saved during meta-testing.
    Returns two numpy arrays, one for the deviations of the gradients and one for the deviations of the updates.
    """
    ### check inputs
    opter_name = opter_cls.__name__
    if not validate_inputs_for_collecting_deviations(opter_name, phase):
        raise ValueError("Invalid inputs")

    ### collect deviations
    rescale_sym_grad_deviations = []
    rescale_sym_update_deviations = []
    for iter_i in range(
        ckpt_iter_freq, n_iters + 1, ckpt_iter_freq
    ):
        if max_iters is not None and iter_i > max_iters:
            break
        ckpt_path = f"{ckpt_path_prefix}{iter_i}.pt"
        if opter_name == "Optimizer":  # L2O
            # load checkpoint
            (
                optee,
                opter,
                optee_grads,
                optee_updates,
                loss_history,
            ) = load_l2o_opter_ckpt(
                path=ckpt_path,
                optee_cls=optee_cls,
                opter_cls=opter_cls,
                optee_config=optee_config,
                opter_config=opter_config,
            )
        else:
            # load checkpoint
            optee, opter, optee_grads, loss_history = load_baseline_opter_ckpt(
                path=ckpt_path,
                optee_cls=optee_cls,
                opter_cls=opter_cls,
                optee_config=optee_config,
                opter_config=opter_config,
            )
            # calculate optee updates
            optee_updates = get_baseline_opter_param_updates(optee, opter)

        # calculate deviations
        rescale_sym_grad_deviations.append(
            get_rescale_sym_constraint_deviation(
                W1=optee.layers.mat_0.weight.cpu(),
                b=optee.layers.mat_0.bias.cpu(),
                W2=optee.layers.final_mat.weight.cpu(),
                W1_update=optee_grads["layers.mat_0.weight"].cpu(),
                b_update=optee_grads["layers.mat_0.bias"].cpu(),
                W2_update=optee_grads["layers.final_mat.weight"].cpu(),
            ).item()
        )
        rescale_sym_update_deviations.append(
            get_rescale_sym_constraint_deviation(
                W1=optee.layers.mat_0.weight.cpu(),
                b=optee.layers.mat_0.bias.cpu(),
                W2=optee.layers.final_mat.weight.cpu(),
                W1_update=-1 * optee_updates["layers.mat_0.weight"].cpu(),
                b_update=-1 * optee_updates["layers.mat_0.bias"].cpu(),
                W2_update=-1 * optee_updates["layers.final_mat.weight"].cpu(),
            ).item()
        )

    rescale_sym_grad_deviations = np.array(rescale_sym_grad_deviations)
    rescale_sym_update_deviations = np.array(rescale_sym_update_deviations)
    return rescale_sym_grad_deviations, rescale_sym_update_deviations


def collect_translation_sym_deviations(
    ckpt_iter_freq,
    n_iters,
    optee_cls,
    opter_cls,
    optee_config=None,
    opter_config=None,
    phase="meta_testing",
    ckpt_path_prefix="",
    max_iters=None,
):
    """
    Collects get_translation_sym_constraint_deviations() for all checkpoints saved during the given phase.
    Returns two numpy arrays, one for the deviations of the gradients and one for the deviations of the updates.
    """
    ### check inputs
    opter_name = opter_cls.__name__
    if not validate_inputs_for_collecting_deviations(opter_name, phase):
        raise ValueError("Invalid inputs")

    ### collect deviations
    tranlation_sym_grad_deviations = []
    tranlation_sym_update_deviations = []
    for iter_i in range(
        ckpt_iter_freq, n_iters + 1, ckpt_iter_freq
    ):
        if max_iters is not None and iter_i > max_iters:
            break
        ckpt_path = f"{ckpt_path_prefix}{iter_i}.pt"
        if opter_name == "Optimizer":  # L2O
            # load checkpoint
            (
                optee,
                opter,
                optee_grads,
                optee_updates,
                loss_history,
            ) = load_l2o_opter_ckpt(
                path=ckpt_path,
                optee_cls=optee_cls,
                opter_cls=opter_cls,
                optee_config=optee_config,
                opter_config=opter_config,
            )
        else:
            # load checkpoint
            optee, opter, optee_grads, loss_history = load_baseline_opter_ckpt(
                path=ckpt_path,
                optee_cls=optee_cls,
                opter_cls=opter_cls,
                optee_config=optee_config,
                opter_config=opter_config,
            )
            # calculate optee updates
            optee_updates = get_baseline_opter_param_updates(optee, opter)

        ### calcualte deviations
        sym_constraint_devs = get_translation_sym_constraint_deviations(
            W_update=optee_grads["layers.final_mat.weight"].cpu(),
            b_update=optee_grads["layers.final_mat.bias"].cpu(),
        )
        tranlation_sym_grad_deviations.append([d.item() for d in sym_constraint_devs])

        sym_constraint_devs = get_translation_sym_constraint_deviations(
            W_update=-1 * optee_updates["layers.final_mat.weight"].cpu(),
            b_update=-1 * optee_updates["layers.final_mat.bias"].cpu(),
        )
        tranlation_sym_update_deviations.append([d.item() for d in sym_constraint_devs])

    tranlation_sym_grad_deviations = np.array(tranlation_sym_grad_deviations)
    tranlation_sym_update_deviations = np.array(tranlation_sym_update_deviations)
    return tranlation_sym_grad_deviations, tranlation_sym_update_deviations


def collect_scale_sym_deviations(
    ckpt_iter_freq,
    n_iters,
    optee_cls,
    opter_cls,
    optee_config=None,
    opter_config=None,
    phase="meta_testing",
    ckpt_path_prefix="",
    max_iters=None,
):
    """
    Collects get_scale_sym_constraint_deviation() for all checkpoints saved during meta-testing.
    Returns two numpy arrays, one for the deviations of the gradients and one for the deviations of the updates.
    """
    ### check inputs
    opter_name = opter_cls.__name__
    if not validate_inputs_for_collecting_deviations(opter_name, phase):
        raise ValueError("Invalid inputs")

    ### collect deviations
    scale_sym_grad_deviations = []
    scale_sym_update_deviations = []
    for iter_i in range(
        ckpt_iter_freq, n_iters + 1, ckpt_iter_freq
    ):
        if max_iters is not None and iter_i > max_iters:
            break
        ckpt_path = f"{ckpt_path_prefix}{iter_i}.pt"
        if opter_name == "Optimizer":  # L2O
            # load checkpoint
            (
                optee,
                opter,
                optee_grads,
                optee_updates,
                loss_history,
            ) = load_l2o_opter_ckpt(
                path=ckpt_path,
                optee_cls=optee_cls,
                opter_cls=opter_cls,
                optee_config=optee_config,
                opter_config=opter_config,
            )
        else:
            # load checkpoint
            optee, opter, optee_grads, loss_history = load_baseline_opter_ckpt(
                path=ckpt_path,
                optee_cls=optee_cls,
                opter_cls=opter_cls,
                optee_config=optee_config,
                opter_config=opter_config,
            )
            # calculate optee updates
            optee_updates = get_baseline_opter_param_updates(optee, opter)

        # calculate deviations
        scale_sym_grad_deviations.append(
            get_scale_sym_constraint_deviation(
                W=optee.layers.mat_0.weight.cpu(),
                b=optee.layers.mat_0.bias.cpu(),
                W_update=optee_grads["layers.mat_0.weight"].cpu(),
                b_update=optee_grads["layers.mat_0.bias"].cpu(),
            ).item()
        )
        scale_sym_update_deviations.append(
            get_scale_sym_constraint_deviation(
                W=optee.layers.mat_0.weight.cpu(),
                b=optee.layers.mat_0.bias.cpu(),
                W_update=-1 * optee_updates["layers.mat_0.weight"].cpu(),
                b_update=-1 * optee_updates["layers.mat_0.bias"].cpu(),
            ).item()
        )

    scale_sym_grad_deviations = np.array(scale_sym_grad_deviations)
    scale_sym_update_deviations = np.array(scale_sym_update_deviations)
    return scale_sym_grad_deviations, scale_sym_update_deviations


def collect_conservation_law_deviations(func, opter_cls, opter_config, optee_cls, optee_config,
    ckpt_iter_freq, n_iters, ckpt_path_prefix, is_l2o=True, max_iters=None):
    ### collect deviations
    conservation_law_devs = []
    params_t0 = None
    for iter_i in range(
        ckpt_iter_freq,
        n_iters + 1,
        ckpt_iter_freq,
    ):
        if max_iters is not None and iter_i > max_iters:
            break
        ### load checkpoint
        ckpt_path = f"{ckpt_path_prefix}{iter_i}.pt"
        ckpt = torch.load(ckpt_path)
        optee = w(optee_cls(**optee_config if optee_config is not None else {}))
        optee.load_state_dict(ckpt["optimizee"])

        ### set params_t0
        if iter_i == ckpt_iter_freq and params_t0 is None:
            params_t0 = dict()
            for n, p in optee.all_named_parameters():
                params_t0[n] = p.clone().detach()
        
        ### calculate deviations
        conservation_law_devs.append(
            func(
                optee=optee,
                params_t0=params_t0
            ).item()
        )
    return conservation_law_devs


def calc_sai(vec_t0, vec_t1, time_delta=1, normalize=True):
    """ Calculate Stiffness-Aware Index (SAI) from the state transition. """
    state_change = vec_t1 - vec_t0
    state_change = torch.norm(state_change / time_delta, p=2)

    if normalize:
        norm_factor = 1 / (torch.norm(vec_t0, p=2) + 1e-8)
        state_change = state_change * norm_factor

    return state_change
