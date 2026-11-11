"""Encoder-once runtime, with validation before advancing the encoder timeline."""
from __future__ import annotations

import math
import torch


class V25Runtime:
    def __init__(self, radar_encoder, controller):
        self.radar_encoder,self.controller = radar_encoder,controller
        self.encoder_state = None

    def reset(self):
        self.encoder_state = None
        self.controller.reset()

    @torch.inference_mode()
    def step(self,rd,md,sensor_index,frame_mask,timestamp,metadata=None):
        previous = self.controller._previous_frame
        if not math.isfinite(timestamp) or (previous is not None and timestamp <= previous.timestamp):
            raise ValueError('runtime timestamps must be finite and strictly increasing')
        self.controller.poll()
        if self.controller.failures:
            raise RuntimeError('; '.join(self.controller.failures))
        encoded = self.radar_encoder.forward_update(rd,md,sensor_index=sensor_index,
                                                    state=self.encoder_state,frame_mask=frame_mask)
        result = self.controller.step_tokens(encoded.radar_tokens,encoded.radar_token_mask,timestamp,metadata)
        self.encoder_state = encoded.state
        return result
