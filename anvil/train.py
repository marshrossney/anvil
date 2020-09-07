"""
train.py

Module containing functions required to train model
"""

from tqdm import tqdm

import torch
import torch.optim as optim

from anvil.utils import get_num_parameters

import logging

log = logging.getLogger(__name__)


def shifted_kl(
    model_log_density: torch.Tensor, target_log_density: torch.Tensor
) -> torch.Tensor:
    r"""Sample mean of the shifted Kullbach-Leibler divergence between target
    and model distribution.

    Parameters
    ----------
    model_log_density: torch.Tensor
        column of log (\tilde p) for a sample of states, which is returned by
        forward pass of `RealNVP` model
    target_log_density: torch.Tensor
        column of log(p) for a sample of states, which includes the negative
        action -S(\phi) and a possible contribution to the volume element due
        to a non-trivial parameterisation.

    Returns
    -------
    out: torch.Tensor
        torch tensor with single element, corresponding to the estimation of the
        shifted K-L for set of sample states.

    """
    return torch.mean(model_log_density - target_log_density, dim=0)


def train(
    loaded_model,
    base_dist,
    target_dist,
    *,
    train_range,
    save_interval,
    n_batch,
    outpath,
    current_loss,
    loaded_optimizer,
    loaded_scheduler,
    increase_batch=False,
    decrease_lr=False,
):
    """training loop of model"""

    num_parameters = get_num_parameters(loaded_model)
    log.info(f"Model has {num_parameters} trainable parameters.")

    if increase_batch is not False:
        ib_epoch, ib_add = increase_batch
        increase_batch = True
    if decrease_lr is not False:
        dlr_epoch, dlr_mult = decrease_lr
        decrease_lr = True

    # let's use tqdm to see progress
    pbar = tqdm(range(*train_range), desc=f"loss: {current_loss}")
    for i in pbar:
        if (i % save_interval) == 0:
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": loaded_model.state_dict(),
                    "optimizer_state_dict": loaded_optimizer.state_dict(),
                    "loss": current_loss,
                },
                f"{outpath}/checkpoint_{i}.pt",
            )
        # gen simple states (gradients not tracked)
        x, base_log_density = base_dist(n_batch)

        sign = x.sum(dim=1).sign()
        neg = (sign < 0).nonzero().squeeze()

        # apply inverse map, calc log density of forward map (gradients tracked)
        phi, model_log_density = loaded_model(x, base_log_density, neg)

        # compute loss function (gradients tracked)
        target_log_density = target_dist.log_density(phi)
        current_loss = shifted_kl(model_log_density, target_log_density)

        # backprop and step model parameters
        loaded_optimizer.zero_grad()  # zero gradients from prev minibatch
        current_loss.backward()  # accumulate new gradients
        loaded_optimizer.step()
        
        #loaded_scheduler.step(current_loss)
        loaded_scheduler.step()
        
        if (i % 50) == 0:
            pbar.set_description(f"loss: {current_loss.item()}")
            with open("loss.txt", "a") as f:
                f.write(f"{float(current_loss)}\n")

        if increase_batch and i == ib_epoch:
            n_batch += ib_add
            log.info(f"increasing batch size to {n_batch}")
        if decrease_lr and i == dlr_epoch:
            loaded_scheduler.base_lrs[0] *= dlr_mult
            log.info(f"decreasing max lr to {loaded_scheduler.base_lrs[0]}")
        
    torch.save(
        {
            "epoch": train_range[-1],
            "model_state_dict": loaded_model.state_dict(),
            "optimizer_state_dict": loaded_optimizer.state_dict(),
            "loss": current_loss,
        },
        f"{outpath}/checkpoint_{train_range[-1]}.pt",
    )

    return loaded_model


def adam(
    loaded_model,
    loaded_checkpoint,
    *,
    learning_rate=0.001,
    adam_betas=(0.9, 0.999),
    optimizer_stability_factor=1e-08,
    optimizer_weight_decay=0,
    adam_use_amsgrad=False,
):
    optimizer = optim.Adam(
        loaded_model.parameters(),
        lr=learning_rate,
        betas=adam_betas,
        eps=optimizer_stability_factor,
        weight_decay=optimizer_weight_decay,
        amsgrad=adam_use_amsgrad,
    )
    if loaded_checkpoint is not None:
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    return optimizer


def adamw(
    loaded_model,
    loaded_checkpoint,
    *,
    learning_rate=0.001,
    adam_betas=(0.9, 0.999),
    optimizer_stability_factor=1e-08,
    optimizer_weight_decay=0,
    adam_use_amsgrad=False,
):
    optimizer = optim.AdamW(
        loaded_model.parameters(),
        lr=learning_rate,
        betas=adam_betas,
        eps=optimizer_stability_factor,
        weight_decay=optimizer_weight_decay,
        amsgrad=adam_use_amsgrad,
    )
    if loaded_checkpoint is not None:
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    return optimizer


def adadelta(
    loaded_model,
    loaded_checkpoint,
    *,
    learning_rate=1.0,
    adadelta_rho=0.9,
    optimizer_stability_factor=1e-06,
    optimizer_weight_decay=0,
):
    optimizer = optim.Adadelta(
        loaded_model.parameters(),
        lr=learning_rate,
        rho=adadelta_rho,
        eps=optimizer_stability_factor,
        weight_decay=optimizer_weight_decay,
    )
    if loaded_checkpoint is not None:
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    return optimizer


def stochastic_gradient_descent(
    loaded_model,
    loaded_checkpoint,
    *,
    learning_rate,
    optimizer_momentum=0,
    optimizer_dampening=0,
    optimizer_weight_decay=0,
    sgd_use_nesterov=False,
):
    optimizer = optim.SGD(
        loaded_model.parameters(),
        lr=learning_rate,
        momentum=optimizer_momentum,
        dampening=optimizer_dampening,
        weight_decay=optimizer_weight_decay,
        nesterov=sgd_use_nesterov,
    )
    if loaded_checkpoint is not None:
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    return optimizer


def rms_prop(
    loaded_model,
    loaded_checkpoint,
    *,
    learning_rate=0.01,
    rmsprop_smoothing=0.99,
    optimizer_stability_factor=1e-08,
    optimizer_weight_decay=0,
    optimizer_momentum=0,
    rmsprop_use_centered=False,
):
    optimizer = optim.RMSprop(
        loaded_model.parameters(),
        lr=learning_Rate,
        alpha=rmsprop_smoothing,
        eps=optimizer_stability_factor,
        weight_decay=optimizer_weight_decay,
        momentum=optimizer_momentum,
        centered=rmsprop_use_centered,
    )
    if loaded_checkpoint is not None:
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    return optimizer


def reduce_lr_on_plateau(
    loaded_optimizer,
    *,
    lr_reduction_factor=0.1,
    min_learning_rate=0,
    patience=500,  # not the PyTorch default
    cooldown=0,
    verbose_scheduler=True,
    scheduler_threshold=0.0001,
    scheduler_threshold_mode="rel",
    scheduler_stability_factor=1e-08,
):
    return optim.lr_scheduler.ReduceLROnPlateau(
        loaded_optimizer,
        factor=lr_reduction_factor,
        patience=patience,
        verbose=verbose_scheduler,
        threshold=scheduler_threshold,
        threshold_mode=scheduler_threshold_mode,
        cooldown=cooldown,
        min_lr=min_learning_rate,
        eps=scheduler_stability_factor,
    )

def cosine_annealing_warm_restarts(
    loaded_optimizer,
    *,
    T_0=1000,
    T_mult=2,
    eta_min=0
):
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(
        loaded_optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
    )


OPTIMIZER_OPTIONS = {
    "adam": adam,
    "adamw": adamw,
    "adadelta": adadelta,
    "sgd": stochastic_gradient_descent,
    "rms_prop": rms_prop,
}

SCHEDULER_OPTIONS = {
    "rlrop": reduce_lr_on_plateau,
    "warm": cosine_annealing_warm_restarts,
}
