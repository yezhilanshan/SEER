"""Per-coordinate Gaussian NLL and explicit recording normalization."""
import math
import torch


def gaussian_nll(mean, logvar, target):
    mean, logvar, target = mean.float(), logvar.float(), target.detach().float()
    if mean.shape != target.shape or logvar.shape != target.shape:
        raise ValueError('Gaussian tensors must match')
    if not all(torch.isfinite(x).all() for x in (mean,logvar,target)):
        raise ValueError('Gaussian targets and predictions must be finite')
    return .5*(math.log(2*math.pi)+logvar+(target-mean).square()*torch.exp(-logvar)).mean(-1)


def reduce_prediction_losses(losses, recording_ids, valid):
    if not valid.any(): raise ValueError('no valid prediction loss')
    return torch.stack([losses[valid & (recording_ids == key)].mean()
                        for key in torch.unique(recording_ids[valid])]).mean()


def fit_persistence_variance(fit_current, fit_future, recording_ids=None):
    if len(fit_current) == 0 or fit_current.shape != fit_future.shape:
        raise ValueError('fit residuals required')
    residual = (fit_future-fit_current).double()
    if not torch.isfinite(residual).all(): raise ValueError('finite fit residuals required')
    if recording_ids is None: recording_ids=torch.zeros(len(residual),dtype=torch.long)
    if recording_ids.shape != (len(residual),): raise ValueError('recording IDs must match residuals')
    second=torch.stack([residual[recording_ids==key].square().mean(0) for key in torch.unique(recording_ids)]).mean(0)
    return second.clamp(min=math.exp(-6),max=math.exp(2)).log().float()
