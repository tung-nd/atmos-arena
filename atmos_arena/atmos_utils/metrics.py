# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import torchist
from scipy import stats
from sklearn.metrics import r2_score

def mse(pred, y, vars, lat=None, mask=None):
    """Mean squared error

    Args:
        pred: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """

    loss = (pred - y) ** 2

    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (loss[:, i] * mask).sum() / mask.sum()
            else:
                loss_dict[var] = loss[:, i].mean()

    if mask is not None:
        loss_dict["loss"] = (loss.mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = loss.mean(dim=1).mean()

    return loss_dict


def lat_weighted_mse(pred, y, vars, lat, mask=None):
    """Latitude weighted mean squared error

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [N, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[var] = (error[:, i] * w_lat).mean()

    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()

    return loss_dict


def lat_weighted_mse_val(pred, y, transform, vars, lat, clim, log_postfix="", mask=None):
    """Latitude weighted mean squared error
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
        mask: 1 for masked values, 0 for visible values
    """

    error = (pred - y) ** 2  # [B, V, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[f"w_mse_{var}{log_postfix}"] = (error[:, i] * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[f"w_mse_{var}{log_postfix}"] = (error[:, i] * w_lat).mean()

    if mask is not None:
        loss_dict["w_mse"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["w_mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_rmse(pred, y, transform, vars, lat, clim, log_postfix="", mask=None):
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
        mask: 1 for masked values, 0 for visible values
    """

    pred = transform(pred)
    y = transform(y)

    error = (pred - y) ** 2  # [B, V, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[f"w_rmse_{var}{log_postfix}"] = torch.mean(
                    torch.sqrt(torch.sum(error[:, i] * w_lat * mask, dim=(-2, -1)) / mask.sum(dim=(-2, -1)))
                )
            else:
                loss_dict[f"w_rmse_{var}{log_postfix}"] = torch.mean(
                    torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
                )
                

    loss_dict["w_rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_acc(pred, y, transform, vars, lat, clim, log_postfix=""):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    # clim = torch.mean(y, dim=(0, 1), keepdim=True)
    # clim = clim.to(device=y.device).unsqueeze(0)
    pred = pred - clim
    y = y - clim
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_prime = pred[:, i] - torch.mean(pred[:, i])
            y_prime = y[:, i] - torch.mean(y[:, i])
            loss_dict[f"acc_{var}{log_postfix}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
                torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
            )

    loss_dict["acc"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

def spectral_div(pred, y, transform, vars, lat, clim, log_postfix="", percentile=0.9):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)
    loss_dict = {}
    
    nx, ny = pred.shape[-2:]
    kx = torch.fft.fftfreq(nx) * nx
    ky = torch.fft.fftfreq(ny) * ny
    kx, ky = torch.meshgrid(kx, ky)
    
    k = torch.sqrt(kx**2 + ky**2).reshape(-1).to(pred.device)
    k_low = 0.5
    k_upp = torch.max(k)
    k_nbin = torch.arange(k_low, torch.max(k), 1).size(0)
    
    # Get percentile index
    k_percentile_idx = int(k_nbin * percentile)

    with torch.no_grad():
        for i, var in enumerate(vars):
            predictions = pred[:, i]
            targets = y[:, i]
            
            predictions = predictions.reshape(predictions.shape[0], -1, predictions.shape[-2], predictions.shape[-1])
            targets = targets.reshape(targets.shape[0], -1, targets.shape[-2], targets.shape[-1])
            
            assert predictions.shape[1] == targets.shape[1]
            nc = predictions.shape[1]
            
            # Handling missing values in predictions
            pred_means = torch.nanmean(predictions, dim=(-2, -1), keepdim=True)
            predictions = torch.where(torch.isnan(predictions), pred_means, predictions)
            
            # Compute along mini-batch
            predictions, targets = torch.nanmean(predictions, dim=0), torch.nanmean(targets, dim=0)
            
            # Transform prediction and targets onto the Fourier space and compute the power
            predictions_power, targets_power = torch.fft.fft2(predictions), torch.fft.fft2(targets)
            predictions_power, targets_power = torch.abs(predictions_power)**2, torch.abs(targets_power)**2
            
            
            
            
            predictions_Sk = torchist.histogram(k.repeat(nc), k_nbin, k_low, k_upp, weights=predictions_power) \
                            / torchist.histogram(k.repeat(nc), k_nbin, k_low, k_upp)

            targets_Sk = torchist.histogram(k.repeat(nc), k_nbin, k_low, k_upp, weights=targets_power) \
                        / torchist.histogram(k.repeat(nc), k_nbin, k_low, k_upp)
            
            # Extract top-k percentile wavenumber and its corresponding power spectrum
            predictions_Sk = predictions_Sk[k_percentile_idx:]
            targets_Sk = targets_Sk[k_percentile_idx:]
            
            # Normalize as pdf along ordered k
            predictions_Sk = predictions_Sk / torch.nansum(predictions_Sk)
            targets_Sk = targets_Sk / torch.nansum(targets_Sk)
            
            
            div = torch.nansum(targets_Sk * torch.log(torch.clamp(targets_Sk / predictions_Sk, min=1e-9)))
            
            
            loss_dict[f"spectral_div_{var}{log_postfix}"] = div

    return loss_dict


def lat_weighted_nrmses(pred, y, transform, vars, lat, clim, log_postfix=""):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)
    y_normalization = clim

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(-1).to(dtype=y.dtype, device=y.device)  # (H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_ = pred[:, i]  # B, H, W
            y_ = y[:, i]  # B, H, W
            error = (torch.mean(pred_, dim=0) - torch.mean(y_, dim=0)) ** 2  # H, W
            error = torch.mean(error * w_lat)
            loss_dict[f"w_nrmses_{var}"] = torch.sqrt(error) / y_normalization
    
    return loss_dict


def lat_weighted_nrmseg(pred, y, transform, vars, lat, clim, log_postfix=""):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)
    y_normalization = clim

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=y.dtype, device=y.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_ = pred[:, i]  # B, H, W
            pred_ = torch.mean(pred_ * w_lat, dim=(-2, -1))  # B
            y_ = y[:, i]  # B, H, W
            y_ = torch.mean(y_ * w_lat, dim=(-2, -1))  # B
            error = torch.mean((pred_ - y_) ** 2)
            loss_dict[f"w_nrmseg_{var}"] = torch.sqrt(error) / y_normalization

    return loss_dict


def lat_weighted_nrmse(pred, y, transform, vars, lat, clim, log_postfix=""):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    nrmses = lat_weighted_nrmses(pred, y, transform, vars, lat, clim, log_postfix)
    nrmseg = lat_weighted_nrmseg(pred, y, transform, vars, lat, clim, log_postfix)
    loss_dict = {}
    for var in vars:
        loss_dict[f"w_nrmses_{var}"] = nrmses[f"w_nrmses_{var}"]
        loss_dict[f"w_nrmseg_{var}"] = nrmseg[f"w_nrmseg_{var}"]
        loss_dict[f"w_nrmse_{var}"] = nrmses[f"w_nrmses_{var}"] + 5 * nrmseg[f"w_nrmseg_{var}"]
    return loss_dict


def remove_nans(pred: torch.Tensor, gt: torch.Tensor):
    # pred and gt are two flattened arrays
    pred_nan_ids = torch.isnan(pred) | torch.isinf(pred)
    pred = pred[~pred_nan_ids]
    gt = gt[~pred_nan_ids]

    gt_nan_ids = torch.isnan(gt) | torch.isinf(gt)
    pred = pred[~gt_nan_ids]
    gt = gt[~gt_nan_ids]

    return pred, gt


def pearson(pred, y, transform, vars, lat, clim, log_postfix="", mask=None):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    mask: 1 for masked values, 0 for visible values
    """

    pred = transform(pred)
    y = transform(y)
    
    pred = pred - clim
    y = y - clim

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_, y_ = pred[:, i].flatten(), y[:, i].flatten()
            mask = mask.flatten() if mask is not None else None
            # pred_, y_ = remove_nans(pred_, y_)
            if mask is not None:
                mask = mask.to(dtype=torch.bool)
                pred_, y_ = pred_[mask], y_[mask]
            loss_dict[f"pearsonr_{var}{log_postfix}"] = stats.pearsonr(pred_.cpu().numpy(), y_.cpu().numpy())[0]

    loss_dict["pearsonr"] = np.mean([loss_dict[k] for k in loss_dict.keys()])

    return loss_dict


def r2(pred, y, transform, vars, lat, clim, log_postfix="", mask=None):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    mask: 1 for masked values, 0 for visible values
    """

    pred = transform(pred)
    y = transform(y)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_, y_ = pred[:, i].flatten(), y[:, i].flatten()
            mask = mask.flatten() if mask is not None else None
            # pred_, y_ = remove_nans(pred_, y_)
            if mask is not None:
                mask = mask.to(dtype=torch.bool)
                pred_, y_ = pred_[mask], y_[mask]
            loss_dict[f"r2_{var}{log_postfix}"] = r2_score(pred_.cpu().numpy(), y_.cpu().numpy())

    return loss_dict


def mean_bias(pred, y, transform, vars, lat, clim, log_postfix="", mask=None):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    mask: 1 for masked values, 0 for visible values
    """

    pred = transform(pred)
    y = transform(y)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_, y_ = pred[:, i].flatten(), y[:, i].flatten()
            mask = mask.flatten() if mask is not None else None
            # pred_, y_ = remove_nans(pred_, y_)
            if mask is not None:
                mask = mask.to(dtype=torch.bool)
                pred_, y_ = pred_[mask], y_[mask]
            loss_dict[f"mean_bias_{var}{log_postfix}"] = pred_.mean() - y_.mean()

    return loss_dict


def lat_weighted_mean_bias(pred, y, transform, vars, lat, clim, log_postfix=""):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    # # lattitude weights
    # w_lat = np.cos(np.deg2rad(lat))
    # w_lat = w_lat / w_lat.mean()  # (H, )
    # w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_, y_ = pred[:, i].flatten(), y[:, i].flatten()
            pred_, y_ = remove_nans(pred_, y_)
            loss_dict[f"mean_bias_{var}{log_postfix}"] = pred_.mean() - y_.mean()

            # pred_mean = torch.mean(w_lat * pred[:, step - 1, i])
            # y_mean = torch.mean(w_lat * y[:, step - 1, i])
            # loss_dict[f"mean_bias_{var}_day_{day}"] = y_mean - pred_mean

    loss_dict["mean_bias"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict
