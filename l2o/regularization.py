import torch

from l2o.analysis import (get_rescale_sym_constraint_deviation,
                          get_scale_sym_constraint_deviation,
                          get_translation_sym_constraint_deviations)


def regularize_updates_translation_constraints(
    updates, optee, lr=1.0
):
    cons_deviations = get_translation_sym_constraint_deviations(
        W_update=-1 * lr * updates["layers.final_mat.weight"],
        b_update=-1 * lr * updates["layers.final_mat.bias"],
    )
    return torch.abs(cons_deviations[0]) + torch.abs(cons_deviations[1])


def regularize_updates_scale_constraints(
    updates, optee, lr=1.0
):
    return get_scale_sym_constraint_deviation(
        W=optee.layers.mat_0.weight.detach(),
        b=optee.layers.mat_0.bias.detach(),
        W_update=-1 * lr * updates["layers.mat_0.weight"],
        b_update=-1 * lr * updates["layers.mat_0.bias"],
    )


def regularize_updates_rescale_constraints(
    updates, optee, lr=1.0
):
    return get_rescale_sym_constraint_deviation(
        W1=optee.layers.mat_0.weight.detach(),
        b=optee.layers.mat_0.bias.detach(),
        W2=optee.layers.final_mat.weight.detach(),
        W1_update=-1 * lr * updates["layers.mat_0.weight"],
        b_update=-1 * lr * updates["layers.mat_0.bias"],
        W2_update=-1 * lr * updates["layers.final_mat.weight"],
    )


def regularize_updates_constraints(
    updates,
    optee,
    lr=1.0,
    rescale_mul=1 / 3,
    scale_mul=1 / 3,
    translation_mul=1 / 3,
):
    """Combines all regularization functions."""
    kwargs = {
        "updates": updates,
        "optee": optee,
        "lr": lr,
    }
    return (
        regularize_updates_rescale_constraints(**kwargs) * rescale_mul
        + regularize_updates_scale_constraints(**kwargs) * scale_mul
        + regularize_updates_translation_constraints(**kwargs) * translation_mul
    )


def regularize_update_norms(updates, optee, lr=1.0):
    """Regularizes the norm of the updates."""
    return sum([torch.norm(lr * u) for u in updates.values()])



"""
--------
OBSOLETE
--------

def regularize_translation_conservation_law_breaking(optee, params_t0):
    \"""
    The following should hold when trained under the gradient flow:
        inner(params_t, ind_vector) = inner(params_t0, ind_vector)
    This function regularizes the deviation from keeping this equality.
    \"""
    if (
        params_t0["layers.final_mat.weight"].requires_grad
        or params_t0["layers.final_mat.bias"].requires_grad
    ):
        print(
            "[WARNING] Regularizing translation law breaking with initial parameters having .requires_grad = True"
        )

    theta_X_t0 = torch.cat((
        params_t0["layers.final_mat.weight"],
        params_t0["layers.final_mat.bias"].unsqueeze(1),
    ), dim=1)  # (n_neurons_l+1, n_neurons_l + 1)
    theta_X_sum_t0 = theta_X_t0.sum(dim=0)  # (n_neurons_l + 1,)

    theta_X_now = torch.cat((
        optee.layers.final_mat.weight,
        optee.layers.final_mat.bias.unsqueeze(1),
    ), dim=1)  # (n_neurons_l+1, n_neurons_l + 1)
    theta_X_sum_now = theta_X_now.sum(dim=0)  # (n_neurons_l + 1,)

    return torch.abs(theta_X_sum_t0 - theta_X_sum_now).sum()


def regularize_rescale_conservation_law_breaking(optee, params_t0):
    \"""
    The following should hold when trained under the gradient flow:
        norm(params_t.subset_A)^2 - norm(params_t.subset_B)^2 = norm(params_t0.subset_A)^2 - norm(params_t0.subset_B)^2
    This function regularizes the deviation from keeping this equality.
    \"""
    ### time t0 - TODO: precompute
    # squared euclidean norm of each neuron's incoming weights
    theta_X1_t0 = (params_t0["layers.mat_0.weight"] ** 2).sum(dim=1)  # (n_neurons,)
    theta_X1_t0 += params_t0["layers.mat_0.bias"] ** 2
    # squared euclidean norm of each neuron's outgoing weights
    theta_X2_t0 = (params_t0["layers.final_mat.weight"] ** 2).sum(dim=0)  # (n_neurons,)
    t0 = theta_X1_t0 - theta_X2_t0

    ### time t
    # squared euclidean norm of each neuron's incoming weights
    theta_X1_now = (optee.layers.mat_0.weight ** 2).sum(dim=1)  # (n_neurons,)
    theta_X1_now += optee.layers.mat_0.bias ** 2
    # squared euclidean norm of each neuron's outgoing weights
    theta_X2_now = (optee.layers.final_mat.weight ** 2).sum(dim=0)  # (n_neurons,)
    now = theta_X1_now - theta_X2_now

    return torch.abs(now - t0).sum()


def regularize_scale_conservation_law_breaking(optee, params_t0):
    \"""
    The following should hold when trained under the gradient flow:
        norm(params_t)^2 = norm(params_t0)^2
    This function regularizes the deviation from keeping this equality.
    \"""
    raise NotImplementedError
    return torch.abs(
        torch.norm(params_t0["layers.mat_0.weight"]) ** 2
        + torch.norm(params_t0["layers.mat_0.bias"]) ** 2
        - torch.norm(optee.layers.mat_0.weight) ** 2
        - torch.norm(optee.layers.mat_0.bias) ** 2
    )


def regularize_conservation_law_breaking(
    optee, params_t0, rescale_mul=1 / 3, scale_mul=1 / 3, translation_mul=1 / 3
):
    \"""Combines all conservation law breaking regularization functions.\"""
    return (
        regularize_rescale_conservation_law_breaking(optee, params_t0) * rescale_mul
        + regularize_scale_conservation_law_breaking(optee, params_t0) * scale_mul
        + regularize_translation_conservation_law_breaking(optee, params_t0)
        * translation_mul
    )

"""