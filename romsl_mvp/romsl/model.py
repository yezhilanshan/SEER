import torch
from torch import nn
from torch.nn import functional as F


class RadarEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU())
        self.gru = nn.GRU(128, 128, batch_first=True)

    def forward(self, points, point_mask):
        # Sanitize padding before the MLP so NaN padding cannot contaminate gradients.
        points = points.masked_fill(~point_mask[..., None], 0)
        features = self.point_mlp(points)
        features = features.masked_fill(~point_mask[..., None], -torch.inf)
        pooled = features.amax(dim=2)
        pooled = torch.where(point_mask.any(-1)[..., None], pooled, torch.zeros_like(pooled))
        hidden, _ = self.gru(pooled)
        return hidden


class RegressionHead(nn.Sequential):
    def __init__(self, dimensions):
        super().__init__(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, dimensions))


def masked_huber(prediction, target, target_mask, frame_mask, weight=None):
    valid = target_mask & frame_mask[..., None] & torch.isfinite(target)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    errors = F.smooth_l1_loss(safe_prediction, safe_target, reduction='none', beta=1)
    weights = torch.ones_like(errors) if weight is None else weight[..., None].expand_as(errors)
    weights = weights * valid
    return (errors * weights).sum() / weights.sum().clamp_min(1)
