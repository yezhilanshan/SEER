"""Dimensionally consistent Figure 4 interpretation; CNN fidelity remains audited."""
from __future__ import annotations
import torch
from torch import nn


def conv_bn(inp: int, out: int, kernel: int, *, relu: bool = True) -> nn.Sequential:
    layers = [nn.Conv1d(inp, out, kernel, padding=kernel//2), nn.BatchNorm1d(out)]
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, inp: int, out: int):
        super().__init__()
        self.main = nn.Sequential(conv_bn(inp, out, 3), conv_bn(out, out, 3, relu=False))
        self.skip = conv_bn(inp, out, 1, relu=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.main(x) + self.skip(x))


class MDFeatureExtractor(nn.Module):
    """Frame-local CNN with three pools; no time padding ever enters BN.

    Figure's ambiguous second skip is interpreted as output16. This is a
    runnable method reconstruction, not a claim to the author's exact CNN.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(conv_bn(1, 4, 3), nn.MaxPool1d(2),
                                 ResidualBlock(4, 8), nn.MaxPool1d(2),
                                 ResidualBlock(8, 16), nn.MaxPool1d(2),
                                 ResidualBlock(16, 32), conv_bn(32, 1, 1))

    def forward(self, md: torch.Tensor) -> torch.Tensor:
        if md.ndim != 3 or md.shape[-1] != 128 or md.shape[1] == 0 or not torch.isfinite(md).all():
            raise ValueError('finite, nonempty MD [B,T,128] required')
        # population standard deviation; exact same transformation for all entry points
        centered = md - md.mean(-1, keepdim=True)
        x = centered / md.std(-1, correction=0, keepdim=True).clamp_min(1e-6)
        b, t, _ = x.shape
        return self.net(x.reshape(b*t, 1, 128)).reshape(b, t, 16)


class SplitBiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = MDFeatureExtractor()
        self.forward_grus = nn.ModuleList(nn.GRU(16, 16, batch_first=True) for _ in range(3))
        self.backward_grus = nn.ModuleList(nn.GRU(n, 16, batch_first=True) for n in (16, 32, 32))
        self.head = nn.Linear(16, 9)

    def forward_chunk(self, md: torch.Tensor, state=None):
        features = self.cnn(md)
        if state is None:
            state = [None]*3
        if len(state) != 3:
            raise ValueError('one hidden state per forward GRU required')
        levels, new_states = [features], []
        for gru, h in zip(self.forward_grus, state):
            features, h_next = gru(features, h)
            levels.append(features); new_states.append(h_next)
        return self.head(features), torch.cat(levels, -1), tuple(new_states)

    def correct(self, cache: torch.Tensor):
        if cache.ndim != 3 or cache.shape[-1] != 64 or cache.shape[1] == 0:
            raise ValueError('nonempty cache [B,T,64] required')
        levels = cache.split(16, -1)
        backward = None
        for i, gru in enumerate(self.backward_grus):
            inp = levels[0] if i == 0 else torch.cat([levels[i], backward], -1)
            backward, _ = gru(inp.flip(1))  # zero state separately for every window
            backward = backward.flip(1)
        return self.head((levels[3] + backward) * .5)

    def forward_sequence(self, md: torch.Tensor):
        forward, cache, _ = self.forward_chunk(md)
        return {'forward': forward, 'corrected': self.correct(cache), 'cache': cache}


class BiGRUBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = MDFeatureExtractor()
        self.gru = nn.GRU(16, 16, num_layers=3, bidirectional=True, batch_first=True)
        self.head = nn.Linear(32, 9)

    def forward(self, md):
        features, _ = self.gru(self.cnn(md))
        return self.head(features)


def parameter_audit(model: SplitBiGRU) -> dict:
    count = lambda m: sum(p.numel() for p in m.parameters())
    cnn, fwd, bwd, head = (count(m) for m in (model.cnn, model.forward_grus, model.backward_grus, model.head))
    return {'cnn_trainable': cnn, 'cnn_target': 8301, 'cnn_difference': cnn-8301,
            'cnn_buffers': sum(b.numel() for b in model.cnn.buffers()),
            'cnn_exact_paper_match': cnn == 8301, 'cnn_fidelity_status': 'figure_interpretation_unconfirmed',
            'forward_grus': fwd, 'backward_grus': bwd, 'shared_head': head,
            'forward_branch': cnn+fwd+head, 'backward_with_shared_head': bwd+head,
            'unique_split_total': cnn+fwd+bwd+head,
            'baseline_total': cnn+12864+297,
            'paper_counts': {'forward': 13350, 'backward_including_shared_head': 6585, 'baseline': 21462},
            'cnn_layers': [{'name': n, 'shape': list(p.shape), 'parameters': p.numel()}
                           for n, p in model.cnn.named_parameters()]}
