"""Reuse frozen radar primitives, with pair state tied only to recording time."""
import torch
from continuous_mmwave.harrad.streaming import PairBuffer, FrozenRDStream
from .memory import TemporalMemory, MemoryItem


class ContinuousMemoryStream:
    def __init__(self, capacities=(16,16,16), group_size=4):
        self.memory=TemporalMemory(capacities,group_size); self.buffer=PairBuffer(); self.observed_frames=0

    def push(self, cells):
        if cells.ndim!=3 or cells.shape[1:]!=(4,64) or not torch.isfinite(cells).all():
            raise ValueError('finite cells[T,4,64] required')
        cells=cells.detach().cpu().float().clone()
        for values,mask in self.buffer.push(cells):
            start=self.memory.observed_frames
            self.memory.append(MemoryItem(values,mask.float(),start,start+int(mask.sum())))
        self.observed_frames+=len(cells)

    def snapshot(self):
        result=TemporalMemory.restore(self.memory.export())
        if self.buffer.pending is not None:
            values=torch.cat((self.buffer.pending,torch.zeros_like(self.buffer.pending)))
            result.append(MemoryItem(values,torch.tensor([1.,0.]),result.observed_frames,self.observed_frames))
        return result


class RadarMemoryStream:
    """Raw normalized RD path; CNN/GRU and pairing survive query boundaries."""
    def __init__(self, backbone, **memory_settings):
        self.radar=FrozenRDStream(backbone); self.state=None
        self.stream=ContinuousMemoryStream(**memory_settings)

    def push(self, normalized_rd):
        output=self.radar.forward_chunk(normalized_rd[None],state=self.state)
        self.state=output['state']; self.stream.push(output['cells'][0])
        return output

    def snapshot(self): return self.stream.snapshot()
