import torch
from torch import optim


def get_rescale_sym_constraint_deviation(W1, b, W2, W1_update, b_update, W2_update):
    """Applies to ReLU, LeakyReLU, Linear, ...
    inner(W1_update, W1) + inner(b_update, b) - inner(W2_update, W2) = 0

    See rescale symmetry in section 3 of the Neural Mechanics paper
    (https://arxiv.org/abs/2012.04728)

    - Note: The left hand side of the constraint should be zero for gradients,
    which are negative updates in the case of SGD (just scaled by lr), hence
    any other parameter updates (from Adam, L2O optimizer, etc.) need to be
    negated (otherwise the deviations will have the opposite sign).
    """
    return (
        W1_update.flatten() @ W1.flatten()
        + b_update.flatten() @ b.flatten()
        - W2_update.flatten() @ W2.flatten()
    ).item()


def get_translation_sym_constraint_deviations(W_update, b_update):
    """Applies to Softmax
    inner(W_update, all_ones) = inner(b_update, all_ones) = 0

    See translation symmetry in section 3 of the Neural Mechanics paper
    (https://arxiv.org/abs/2012.04728)

    - Note: The left hand side of the constraint should be zero for gradients,
    which are negative updates in the case of SGD (just scaled by lr), hence
    any other parameter updates (from Adam, L2O optimizer, etc.) need to be
    negated (otherwise the deviations will have the opposite sign).
    """
    inner_W_update = W_update.sum()  # <=> W_update.flatten() @ ind_vec
    inner_b_update = b_update.sum()  # <=> b_update.flatten() @ ind_vec
    return inner_W_update.item(), inner_b_update.item()


def get_scale_sym_constraint_deviation(W, b, W_update, b_update):
    """Applies to Batch normalization
    inner(W_update, W) + inner(b_update, b) = 0

    See scale symmetry in section 3 of the Neural Mechanics paper
    (https://arxiv.org/abs/2012.04728)

    - Note: The left hand side of the constraint should be zero for gradients,
    which are negative updates in the case of SGD (just scaled by lr), hence
    any other parameter updates (from Adam, L2O optimizer, etc.) need to be
    negated (otherwise the deviations will have the opposite sign).
    """
    inner_W_update = W_update.flatten() @ W.flatten()
    inner_b_update = b_update.flatten() @ b.flatten()
    return inner_W_update.item() + inner_b_update.item()


def get_adam_param_update(
    param, grad, opter_state, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
):
    beta1, beta2 = betas

    if weight_decay != 0.0:
        grad = grad + weight_decay * param

    biased_first_moment = beta1 * opter_state["exp_avg"] + (1 - beta1) * grad
    biased_second_moment = beta2 * opter_state["exp_avg_sq"] + (1 - beta2) * grad**2

    # bias correction
    bias_correction1 = 1 - beta1 ** (opter_state["step"] + 1)
    bias_correction2 = 1 - beta2 ** (opter_state["step"] + 1)
    bias_corrected_first_moment = biased_first_moment / bias_correction1
    bias_corrected_second_moment = biased_second_moment / bias_correction2

    # update (new_param = old_param + param_update)
    param_update = (
        -lr
        * bias_corrected_first_moment
        / (torch.sqrt(bias_corrected_second_moment) + eps)
    )

    return param_update


def get_baseline_opter_param_updates(optee, opter):
    optee_updates = {}
    opter_state_dict = opter.state_dict()
    if isinstance(opter, optim.Adam):
        for p_i, (n, p) in enumerate(optee.all_named_parameters()):
            assert p.grad is not None, f"p.grad is None for {n}"
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
