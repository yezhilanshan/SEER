"""Read-only feature taps and a radar-independent soft prefix control."""
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence


@torch.no_grad()
def frozen_features(encoder, rd, md, lengths):
    captured = []
    def capture(module, inputs):
        fused, _ = pad_packed_sequence(inputs[0], batch_first=True, total_length=rd.shape[1])
        captured.append(fused)
    hook = encoder.gru.register_forward_pre_hook(capture)
    try:
        hidden, _ = encoder(rd, md, lengths)
    finally:
        hook.remove()
    return captured[0], hidden


class ConstantPrefix(nn.Module):
    def __init__(self, tokens, width):
        super().__init__()
        self.prefix = nn.Parameter(torch.empty(tokens, width))
        nn.init.normal_(self.prefix, std=.02)

    def forward(self, rd, md, lengths):
        return None, self.prefix.unsqueeze(0).expand(len(lengths), -1, -1)


def make_language_prior(model):
    model.requires_grad_(False)
    embedding = model.llm.get_input_embeddings().weight
    model.encoder = ConstantPrefix(model.config.tokens, embedding.shape[1]).to(embedding.device)
    model.projector = nn.Identity()
    return model
