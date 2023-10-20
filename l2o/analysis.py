import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import optim

from l2o.others import load_baseline_opter_ckpt, load_l2o_opter_ckpt, w


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

    - Note: The same result would be obtained by using the following:
        total_dev = 0
        for hidden_neuron_idx in range(optee.layers.mat_0.bias.shape[0]):
            curr_dev = get_rescale_sym_constraint_deviation(
                W1=optee.layers.mat_0.weight.cpu()[hidden_neuron_idx],
                b=optee.layers.mat_0.bias.cpu()[hidden_neuron_idx].view(-1),
                W2=optee.layers.final_mat.weight.cpu()[:, hidden_neuron_idx],
                W1_update=-1 * optee_updates["layers.mat_0.weight"].cpu()[hidden_neuron_idx],
                b_update=-1 * optee_updates["layers.mat_0.bias"].cpu()[hidden_neuron_idx].view(-1),
                W2_update=-1 * optee_updates["layers.final_mat.weight"].cpu()[:, hidden_neuron_idx],
            ).item()
            total_dev += curr_dev
        rescale_sym_update_deviations.append(total_dev)
    """
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
    W_deviations = W_update.flatten() @ W.flatten()
    b_deviations = b_update.flatten() @ b.flatten()
    return W_deviations + b_deviations


def validate_inputs_for_collecting_deviations(opter_name, phase):
    assert opter_name in [
        "Optimizer",
        "SGD",
        "Adam",
        "Lion",
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
    optee_updates_lr,
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
    for iter_i in [1, *range(
        ckpt_iter_freq, n_iters + 1, ckpt_iter_freq
    )]:
        if max_iters is not None and iter_i > max_iters:
            break
        ckpt_path = f"{ckpt_path_prefix}{iter_i}.pt"
        if opter_name == "Optimizer":  # L2O
            ### load checkpoint
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
            
            ### calculate deviations for updates
            rescale_sym_update_deviations.append(
            get_rescale_sym_constraint_deviation(
                W1=optee.layers.mat_0.weight.cpu(),
                b=optee.layers.mat_0.bias.cpu(),
                W2=optee.layers.final_mat.weight.cpu(),
                W1_update=optee_updates_lr * optee_updates["layers.mat_0.weight"].cpu(),
                b_update=optee_updates_lr * optee_updates["layers.mat_0.bias"].cpu(),
                W2_update=optee_updates_lr * optee_updates["layers.final_mat.weight"].cpu(),
            ).item()
        )
        else:
            ### load checkpoint
            optee, opter, optee_grads, loss_history = load_baseline_opter_ckpt(
                path=ckpt_path,
                optee_cls=optee_cls,
                opter_cls=opter_cls,
                optee_config=optee_config,
                opter_config=opter_config,
            )
            ### calculate optee updates
            optee_updates = get_baseline_opter_param_updates(optee, opter)

            ### calculate deviations for updates
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
        
        ### calculate deviations for gradients
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

    rescale_sym_grad_deviations = np.array(rescale_sym_grad_deviations)
    rescale_sym_update_deviations = np.array(rescale_sym_update_deviations)
    return rescale_sym_grad_deviations, rescale_sym_update_deviations


def collect_translation_sym_deviations(
    ckpt_iter_freq,
    n_iters,
    optee_cls,
    opter_cls,
    optee_updates_lr,
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
    for iter_i in [1, *range(
        ckpt_iter_freq, n_iters + 1, ckpt_iter_freq
    )]:
        if max_iters is not None and iter_i > max_iters:
            break
        ckpt_path = f"{ckpt_path_prefix}{iter_i}.pt"
        if opter_name == "Optimizer":  # L2O
            ### load checkpoint
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

            ### calculate deviations for updates
            sym_constraint_devs = get_translation_sym_constraint_deviations(
                W_update=optee_updates_lr * optee_updates["layers.final_mat.weight"].cpu(),
                b_update=optee_updates_lr * optee_updates["layers.final_mat.bias"].cpu(),
            )
            tranlation_sym_update_deviations.append([d.item() for d in sym_constraint_devs])
        else:
            ### load checkpoint
            optee, opter, optee_grads, loss_history = load_baseline_opter_ckpt(
                path=ckpt_path,
                optee_cls=optee_cls,
                opter_cls=opter_cls,
                optee_config=optee_config,
                opter_config=opter_config,
            )
            ### calculate optee updates
            optee_updates = get_baseline_opter_param_updates(optee, opter)

            ### calculate deviations for updates
            sym_constraint_devs = get_translation_sym_constraint_deviations(
                W_update=-1 * optee_updates["layers.final_mat.weight"].cpu(),
                b_update=-1 * optee_updates["layers.final_mat.bias"].cpu(),
            )
            tranlation_sym_update_deviations.append([d.item() for d in sym_constraint_devs])
        
        ### calcualte deviations for gradients
        sym_constraint_devs = get_translation_sym_constraint_deviations(
            W_update=optee_grads["layers.final_mat.weight"].cpu(),
            b_update=optee_grads["layers.final_mat.bias"].cpu(),
        )
        tranlation_sym_grad_deviations.append([d.item() for d in sym_constraint_devs])

    tranlation_sym_grad_deviations = np.array(tranlation_sym_grad_deviations)
    tranlation_sym_update_deviations = np.array(tranlation_sym_update_deviations)
    return tranlation_sym_grad_deviations, tranlation_sym_update_deviations


def collect_scale_sym_deviations(
    ckpt_iter_freq,
    n_iters,
    optee_cls,
    opter_cls,
    optee_updates_lr,
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
    for iter_i in [1, *range(
        ckpt_iter_freq, n_iters + 1, ckpt_iter_freq
    )]:
        if max_iters is not None and iter_i > max_iters:
            break
        ckpt_path = f"{ckpt_path_prefix}{iter_i}.pt"
        if opter_name == "Optimizer":  # L2O
            ### load checkpoint
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

            ### calculate deviations for updates
            scale_sym_update_deviations.append(
                get_scale_sym_constraint_deviation(
                    W=optee.layers.mat_0.weight.cpu(),
                    b=optee.layers.mat_0.bias.cpu(),
                    W_update=optee_updates_lr * optee_updates["layers.mat_0.weight"].cpu(),
                    b_update=optee_updates_lr * optee_updates["layers.mat_0.bias"].cpu(),
                ).item()
            )
        else:
            ### load checkpoint
            optee, opter, optee_grads, loss_history = load_baseline_opter_ckpt(
                path=ckpt_path,
                optee_cls=optee_cls,
                opter_cls=opter_cls,
                optee_config=optee_config,
                opter_config=opter_config,
            )
            
            ### calculate optee updates
            optee_updates = get_baseline_opter_param_updates(optee, opter)

            ### calculate deviations for updates
            scale_sym_update_deviations.append(
                get_scale_sym_constraint_deviation(
                    W=optee.layers.mat_0.weight.cpu(),
                    b=optee.layers.mat_0.bias.cpu(),
                    W_update=-1 * optee_updates["layers.mat_0.weight"].cpu(),
                    b_update=-1 * optee_updates["layers.mat_0.bias"].cpu(),
                ).item()
            )
        
        ### calculate deviations for gradients
        scale_sym_grad_deviations.append(
            get_scale_sym_constraint_deviation(
                W=optee.layers.mat_0.weight.cpu(),
                b=optee.layers.mat_0.bias.cpu(),
                W_update=optee_grads["layers.mat_0.weight"].cpu(),
                b_update=optee_grads["layers.mat_0.bias"].cpu(),
            ).item()
        )

    scale_sym_grad_deviations = np.array(scale_sym_grad_deviations)
    scale_sym_update_deviations = np.array(scale_sym_update_deviations)
    return scale_sym_grad_deviations, scale_sym_update_deviations


def get_baseline_opter_param_updates(optee, opter):
    ### init a mirror optee to get the gradients
    m_optee = deepcopy(optee)

    ### init a mirror opter to get the updates
    m_opter = opter.__class__(m_optee.parameters(), **opter.defaults)
    m_opter.load_state_dict(opter.state_dict())

    ### perform the update
    params_before_update = {n: deepcopy(p.detach()) for n, p in m_optee.all_named_parameters()}
    m_opter.step()

    ### get the updates
    optee_updates = {}
    for n, p in m_optee.all_named_parameters():
        if p.grad is None:
            print(f"[WARNING] p.grad is None for {n}.")
            continue
        optee_updates[n] = p - params_before_update[n]

    return optee_updates


def calc_sai(vec_t0, vec_t1, time_delta=1, normalize=True):
    """ Calculate Stiffness-Aware Index (SAI) from the state transition. """
    state_change = vec_t1 - vec_t0
    state_change = torch.norm(state_change / time_delta, p=2)

    if normalize:
        norm_factor = 1 / (torch.norm(vec_t0, p=2) + 1e-8)
        state_change = state_change * norm_factor

    return state_change



"""
--------
OBSOLETE
--------

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
    first_step = False
    if opter_state_dict["state"] is None or len(opter_state_dict["state"]) == 0: # first step
        first_step = True
        if verbose:
            print(f"[WARNING] opter_state_dict['state'] is None, initializing")
        if isinstance(opter, optim.Adam):
            opter_state_dict["state"] = [{
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p),
                "step": 0,
            } for _, p in optee.all_named_parameters()]
        elif isinstance(opter, Lion):
            opter_state_dict["state"] = [{
                "exp_avg": torch.zeros_like(p),
                "step": 0,
            } for _, p in optee.all_named_parameters()]
        elif isinstance(opter, optim.SGD):
            opter_state_dict["state"] = [{
                "momentum_buffer": torch.zeros_like(p),
            } for _, p in optee.all_named_parameters()]
        else:
            raise NotImplementedError(f"Optimizer {type(opter)} not implemented")

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
        for p_i, (n, p) in enumerate(optee.all_named_parameters()):
            if p.grad is not None:
                assert opter_state_dict["param_groups"][0]["nesterov"] is False, "Not implemented"
                assert opter_state_dict["param_groups"][0]["weight_decay"] == 0, "Not implemented"
                ### SGD full step (including momentum if any)
                if opter_state_dict["param_groups"][0]["momentum"] != 0 and not first_step:
                    optee_updates[n] = -opter_state_dict["param_groups"][0]["lr"] * (
                        p.grad * (1 - opter_state_dict["param_groups"][0]["dampening"])
                        + opter_state_dict["state"][p_i]["momentum_buffer"] * opter_state_dict["param_groups"][0]["momentum"]
                    )
                else:
                    optee_updates[n] = -opter_state_dict["param_groups"][0]["lr"] * p.grad
    elif isinstance(opter, Lion):
        for p_i, (n, p) in enumerate(optee.all_named_parameters()):
            if p.grad is None:
                if verbose:
                    print(f"[WARNING] p.grad is None for {n}, skipping")
                continue
            assert p.shape == p.grad.shape
            assert p.shape == opter_state_dict["state"][p_i]["exp_avg"].shape

            optee_updates[n] = p.detach().clone() # param to modify to find out the update

            ### inplace update
            lion_update_fn(
                p=optee_updates[n],
                grad=p.grad,
                exp_avg=opter_state_dict["state"][p_i]["exp_avg"].detach().clone(),
                lr=opter_state_dict["param_groups"][0]["lr"],
                wd=opter_state_dict["param_groups"][0]["weight_decay"],
                beta1=opter_state_dict["param_groups"][0]["betas"][0],
                beta2=opter_state_dict["param_groups"][0]["betas"][1],
            )
            optee_updates[n] = optee_updates[n] - p.detach() # get update
            assert not torch.allclose(optee_updates[n], p)
    else:
        raise NotImplementedError(f"Optimizer {type(opter)} not implemented")
    return optee_updates

    
def collect_conservation_law_deviations(func, opter_cls, opter_config, optee_cls, optee_config,
    ckpt_iter_freq, n_iters, ckpt_path_prefix, is_l2o=True, max_iters=None):
    raise NotImplementedError("This function is not used anymore.")
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

    
"""