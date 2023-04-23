import torch

from l2o.analysis import (
    get_rescale_sym_constraint_deviation,
    get_scale_sym_constraint_deviation,
    get_translation_sym_constraint_deviations,
)


def regularize_updates_translation_constraints(
    updates, optee, lr=1.0, normalize_updates=False, normalize_params=False
):
    cons_deviations = get_translation_sym_constraint_deviations(
        W_update=-1 * lr * updates["layers.final_mat.weight"],
        b_update=-1 * lr * updates["layers.final_mat.bias"],
        normalize_updates=normalize_updates,
    )
    return torch.abs(cons_deviations[0]) + torch.abs(cons_deviations[1])


def regularize_updates_scale_constraints(
    updates, optee, lr=1.0, normalize_updates=False, normalize_params=False
):
    return get_scale_sym_constraint_deviation(
        W=optee.layers.mat_0.weight.detach(),
        b=optee.layers.mat_0.bias.detach(),
        W_update=-1 * lr * updates["layers.mat_0.weight"],
        b_update=-1 * lr * updates["layers.mat_0.bias"],
        normalize_updates=normalize_updates,
        normalize_params=normalize_params,
    )


def regularize_updates_rescale_constraints(
    updates, optee, lr=1.0, normalize_updates=False, normalize_params=False
):
    return get_rescale_sym_constraint_deviation(
        W1=optee.layers.mat_0.weight.detach(),
        b=optee.layers.mat_0.bias.detach(),
        W2=optee.layers.final_mat.weight.detach(),
        W1_update=-1 * lr * updates["layers.mat_0.weight"],
        b_update=-1 * lr * updates["layers.mat_0.bias"],
        W2_update=-1 * lr * updates["layers.final_mat.weight"],
        normalize_updates=normalize_updates,
        normalize_params=normalize_params,
    )


def regularize_updates_constraints(
    updates,
    optee,
    lr=1.0,
    rescale_mul=1 / 3,
    scale_mul=1 / 3,
    translation_mul=1 / 3,
    normalize_updates=False,
    normalize_params=False,
):
    """Combines all regularization functions."""
    kwargs = {
        "updates": updates,
        "optee": optee,
        "lr": lr,
        "normalize_updates": normalize_updates,
        "normalize_params": normalize_params,
    }
    return (
        regularize_updates_rescale_constraints(**kwargs) * rescale_mul
        + regularize_updates_scale_constraints(**kwargs) * scale_mul
        + regularize_updates_translation_constraints(**kwargs) * translation_mul
    )


def regularize_update_norms(updates, optee, lr=1.0):
    """Regularizes the norm of the updates."""
    return sum([torch.norm(lr * u) for u in updates.values()])


def regularize_translation_conservation_law_breaking(optee, params_t0):
    """
    The following should hold when trained under the gradient flow:
        inner(params_t, ind_vector) = inner(params_t0, ind_vector)
    This function regularizes the deviation from keeping this equality.
    """
    if (
        params_t0["layers.final_mat.weight"].requires_grad
        or params_t0["layers.final_mat.bias"].requires_grad
    ):
        print(
            "[WARNING] Regularizing translation law breaking with initial parameters having .requires_grad = True"
        )
    weight_dev = torch.abs(
        torch.sum(params_t0["layers.final_mat.weight"])
        - torch.sum(optee.layers.final_mat.weight)
    )
    bias_dev = torch.abs(
        torch.sum(params_t0["layers.final_mat.bias"])
        - torch.sum(optee.layers.final_mat.bias)
    )
    return weight_dev + bias_dev


def regularize_rescale_conservation_law_breaking(optee, params_t0):
    """
    The following should hold when trained under the gradient flow:
        norm(params_t.subset_A)^2 - norm(params_t.subset_B)^2 = norm(params_t0.subset_A)^2 - norm(params_t0.subset_B)^2
    This function regularizes the deviation from keeping this equality.
    """
    t0_diff = (
        torch.norm(params_t0["layers.mat_0.weight"]) ** 2
        + torch.norm(params_t0["layers.mat_0.bias"]) ** 2
        - torch.norm(params_t0["layers.final_mat.weight"]) ** 2
    )
    t_diff = (
        torch.norm(optee.layers.mat_0.weight) ** 2
        + torch.norm(optee.layers.mat_0.bias) ** 2
        - torch.norm(optee.layers.final_mat.weight) ** 2
    )
    return torch.abs(t0_diff - t_diff)


def regularize_scale_conservation_law_breaking(optee, params_t0):
    """
    The following should hold when trained under the gradient flow:
        norm(params_t)^2 = norm(params_t0)^2
    This function regularizes the deviation from keeping this equality.
    """
    return torch.abs(
        torch.norm(params_t0["layers.mat_0.weight"]) ** 2
        + torch.norm(params_t0["layers.mat_0.bias"]) ** 2
        - torch.norm(optee.layers.mat_0.weight) ** 2
        - torch.norm(optee.layers.mat_0.bias) ** 2
    )


def regularize_conservation_law_breaking(
    optee, params_t0, rescale_mul=1 / 3, scale_mul=1 / 3, translation_mul=1 / 3
):
    """Combines all conservation law breaking regularization functions."""
    return (
        regularize_rescale_conservation_law_breaking(optee, params_t0) * rescale_mul
        + regularize_scale_conservation_law_breaking(optee, params_t0) * scale_mul
        + regularize_translation_conservation_law_breaking(optee, params_t0)
        * translation_mul
    )
