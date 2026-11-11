"""Engineering defaults for the first experiment, independent of legacy configs."""
from dataclasses import asdict,dataclass
import math


@dataclass
class LatentMemoryConfig:
    manifest_path: str = '/home/xibeidaxue/newdisk2/WJ/HARrad/outputs/event_warmup_20261017_v1/feature-manifest.json'
    manifest_sha256: str = '67fa8e6afd4caf401a27cedc6847d2f7748d8c2ddacaba021ce474663d493e97'
    llm_path: str = '/home/xibeidaxue/SEER/Qwen2.5-0.5B-Instruct'
    adapter_checkpoint: str | None = '/home/xibeidaxue/newdisk2/WJ/HARrad/outputs/event_warmup_20261017_v1/training/best.pt'
    adapter_sha256: str | None = '559aee947bc9a284180cfc7f5eefcbf1039c766fe1aec319a5bc7bcf1cc2fa92'
    fixed_instruction: str = 'Describe the observed human actions in chronological order up to now. Preserve repeated actions and indicate any ongoing action. Do not invent unobserved details.'
    fps: float = 15.
    query_frames: int = 30
    capacities: tuple = (16,16,16)
    group_size: int = 4
    lmt_counts: tuple = (4,4,4)
    rounds: int = 2
    top_k: int = 4
    time_encoding: str = 'log_time'
    max_new_tokens: int = 256
    max_answer_tokens: int = 2048
    seed: int = 42
    learning_rate: float = 1e-4
    weight_decay: float = .01
    grad_clip: float = 1.
    checkpoint_every: int = 10
    gradient_checkpointing: bool = True

    def __post_init__(self):
        integer_values=(self.query_frames,self.group_size,self.top_k,self.max_new_tokens,self.max_answer_tokens,self.checkpoint_every,
                        *self.capacities,*self.lmt_counts)
        if any(type(n) is not int or n<=0 for n in integer_values): raise ValueError('positive integer budgets required')
        if len(self.capacities)!=3 or len(self.lmt_counts)!=3 or self.group_size<2 or min(self.capacities)<self.group_size:
            raise ValueError('three groups/capacities must fit compression group')
        if type(self.rounds) is not int or self.rounds<0 or type(self.seed) is not int or self.seed<0: raise ValueError('invalid rounds/seed')
        if any(not math.isfinite(x) or x<=0 for x in (self.fps,self.learning_rate,self.grad_clip)):
            raise ValueError('positive finite fps/learning rate/gradient clip required')
        if not math.isfinite(self.weight_decay) or self.weight_decay<0: raise ValueError('invalid weight decay')
        if self.time_encoding not in ('none','log_time'): raise ValueError('unsupported time encoding')
        if not self.fixed_instruction.strip(): raise ValueError('fixed instruction cannot be empty')
        if self.adapter_checkpoint and not self.adapter_sha256: raise ValueError('adapter SHA required')

    @classmethod
    def from_dict(cls, values):
        unknown=set(values)-set(cls.__dataclass_fields__)
        if unknown: raise ValueError(f'unknown settings: {sorted(unknown)}')
        return cls(**values)

    def to_dict(self): return asdict(self)
