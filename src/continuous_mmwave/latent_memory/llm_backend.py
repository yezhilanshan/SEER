"""One frozen full causal LM shared by independent group reads and caption decoding."""
import torch
from torch import nn
from torch.nn import functional as F


class FrozenCausalBackend(nn.Module):
    def __init__(self, causal_lm):
        super().__init__()
        if causal_lm.config.model_type!='qwen2': raise ValueError('first backend supports Qwen2/Qwen2.5')
        self.lm=causal_lm.requires_grad_(False).eval()
        self.hidden_size=int(causal_lm.config.hidden_size)
        self.context_limit=int(causal_lm.config.max_position_embeddings)
        self.checkpoint_reads=False
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @property
    def device(self): return self.lm.get_input_embeddings().weight.device

    @property
    def dtype(self): return self.lm.get_input_embeddings().weight.dtype

    def embed(self, ids):
        if ids.ndim!=2 or ids.shape[0]!=1 or not ids.shape[1]: raise ValueError('nonempty single-stream ids required')
        return self.lm.get_input_embeddings()(ids.to(self.device))

    def _check(self, length):
        if length>self.context_limit: raise ValueError(f'context budget exceeded: {length}>{self.context_limit}')

    def read(self, prompt_ids, evidence, latents):
        """Separate calls implement strict group visibility without indirect cross-group paths."""
        prompt=self.embed(prompt_ids)
        if evidence.ndim!=2 or latents.ndim!=2 or evidence.shape[-1]!=self.hidden_size or latents.shape[-1]!=self.hidden_size:
            raise ValueError('invalid evidence/LMT dimensions')
        hidden=torch.cat((prompt,evidence[None].to(prompt),latents[None].to(prompt)),1)
        self._check(hidden.shape[1])
        output=self._core_hidden(hidden)
        return output[0,-len(latents):].float()

    def _core_hidden(self, hidden):
        def forward(values):
            return self.lm.model(inputs_embeds=values,
                attention_mask=torch.ones(values.shape[:2],device=self.device,dtype=torch.long),
                use_cache=False,return_dict=True).last_hidden_state
        if self.checkpoint_reads and torch.is_grad_enabled() and hidden.requires_grad:
            from torch.utils.checkpoint import checkpoint
            output=checkpoint(forward,hidden,use_reentrant=False)
        else:
            output=forward(hidden)
        return output

    def answer_loss(self, prefix, prompt_ids, answer_ids):
        if answer_ids.ndim!=2 or answer_ids.shape[0]!=1 or not answer_ids.shape[1]: raise ValueError('nonempty answer required')
        prompt=self.embed(prompt_ids); answer=self.embed(answer_ids)
        hidden=torch.cat((prefix.to(prompt),prompt,answer),1); self._check(hidden.shape[1])
        start=prefix.shape[1]+prompt.shape[1]-1
        selected=self._core_hidden(hidden)[:,start:start+answer.shape[1]].reshape(-1,self.hidden_size)
        targets=answer_ids.to(self.device).reshape(-1); pieces=[]
        head=self.lm.get_output_embeddings()
        def token_loss(values, labels):
            return F.cross_entropy(head(values).float(),labels,reduction='sum')
        # Only answer positions need vocabulary logits. Checkpoint each small head/CE block
        # so long descriptions do not retain a full [prefix+answer,vocabulary] graph.
        for first in range(0,len(targets),64):
            values=selected[first:first+64]; labels=targets[first:first+64]
            if self.checkpoint_reads and torch.is_grad_enabled() and values.requires_grad:
                from torch.utils.checkpoint import checkpoint
                pieces.append(checkpoint(token_loss,values,labels,use_reentrant=False))
            else:
                pieces.append(token_loss(values,labels))
        return torch.stack(pieces).sum()/len(targets)

    @torch.no_grad()
    def generate(self, prefix, prompt_ids, tokenizer, max_new_tokens=256):
        prompt=self.embed(prompt_ids); hidden=torch.cat((prefix.to(prompt),prompt),1)
        self._check(hidden.shape[1]+max_new_tokens)
        generated=self.lm.generate(inputs_embeds=hidden,attention_mask=torch.ones(hidden.shape[:2],device=self.device,dtype=torch.long),
                                   do_sample=False,max_new_tokens=max_new_tokens,use_cache=True,
                                   pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        ids=generated[0].tolist()
        return dict(text=tokenizer.decode(ids,skip_special_tokens=True),token_ids=ids,
                    truncated=bool(len(ids)>=max_new_tokens and (not ids or ids[-1]!=tokenizer.eos_token_id)))
