import torch

from analysis import (
    get_rescale_sym_constraint_deviation,
    get_scale_sym_constraint_deviation,
    get_translation_sym_constraint_deviations,
)


def regularize_updates_translation_constraints(updates, optee, lr=1.0):
    cons_deviations = get_translation_sym_constraint_deviations(
        W_update=-1 * lr * updates["layers.final_mat.weight"],
        b_update=-1 * lr * updates["layers.final_mat.bias"],
    )
    return torch.abs(cons_deviations[0]) + torch.abs(cons_deviations[1])


def regularize_updates_scale_constraints(updates, optee, lr=1.0):
    return get_scale_sym_constraint_deviation(
        W=optee.layers.mat_0.weight.detach(),
        b=optee.layers.mat_0.bias.detach(),
        W_update=-1 * lr * updates["layers.mat_0.weight"],
        b_update=-1 * lr * updates["layers.mat_0.bias"],
    )


def regularize_updates_rescale_constraints(updates, optee, lr=1.0):
    return get_rescale_sym_constraint_deviation(
        W1=optee.layers.mat_0.weight.detach(),
        b=optee.layers.mat_0.bias.detach(),
        W2=optee.layers.final_mat.weight.detach(),
        W1_update=-1 * lr * updates["layers.mat_0.weight"],
        b_update=-1 * lr * updates["layers.mat_0.bias"],
        W2_update=-1 * lr * updates["layers.final_mat.weight"],
    )


def regularize_updates_constraints(
    updates, optee, lr=1.0, rescale_mul=1 / 3, scale_mul=1 / 3, translation_mul=1 / 3
):
    """Combines all regularization functions."""
    return (
        regularize_updates_rescale_constraints(updates, optee, lr=lr) * rescale_mul
        + regularize_updates_scale_constraints(updates, optee, lr=lr) * scale_mul
        + regularize_updates_translation_constraints(updates, optee, lr=lr)
        * translation_mul
    )
