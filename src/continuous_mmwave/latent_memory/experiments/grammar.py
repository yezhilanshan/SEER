"""A cyclic token automaton limits syntax, never the correct class/order/count."""
import torch
from ..data import PHRASES


class DescriptionGrammar:
    def __init__(self, tokenizer):
        self.tokenizer=tokenizer; self.edges=[]; self.epsilon=[]
        self.start=self._node(); tail=self._node(); self.final=self._node()
        for phrase in PHRASES:
            for suffix in ('',' (ongoing)'):
                self._add(self.start,self._encode('The person '+phrase+suffix),tail)
                self._add(tail,self._encode(', then '+phrase+suffix),tail)
        self._add(tail,self._encode('.')+[tokenizer.eos_token_id],self.final)
        self._add(self.start,self._encode('No annotated action has been observed yet.')+[tokenizer.eos_token_id],self.final)

    def _encode(self,text): return self.tokenizer.encode(text,add_special_tokens=False)

    def _node(self):
        self.edges.append({}); self.epsilon.append(set()); return len(self.edges)-1

    def _add(self,start,tokens,end):
        node=start
        for token in tokens:
            if token not in self.edges[node]: self.edges[node][token]=self._node()
            node=self.edges[node][token]
        self.epsilon[node].add(end)

    def _closure(self,states):
        states=set(states); todo=list(states)
        while todo:
            for target in self.epsilon[todo.pop()]:
                if target not in states: states.add(target); todo.append(target)
        return states

    def _walk(self,ids):
        states=self._closure({self.start})
        for token in ids:
            states=self._closure({self.edges[s][int(token)] for s in states if int(token) in self.edges[s]})
            if not states: raise ValueError('generated prefix outside description grammar')
        return states

    def allowed(self,ids):
        tokens=sorted({token for s in self._walk(ids) for token in self.edges[s]})
        # Finished batches can be padded by HF after EOS while other sequences run.
        return tokens or [self.tokenizer.eos_token_id]

    def complete(self,ids): return self.final in self._walk(ids)

    def answer_ids(self, reference):
        seq=reference['sequence']; flags=reference['ongoing']
        if not seq: return self._encode('No annotated action has been observed yet.')+[self.tokenizer.eos_token_id]
        ids=[]
        for i,(label,active) in enumerate(zip(seq,flags)):
            ids+=self._encode(('The person ' if i==0 else ', then ')+PHRASES[label]+(' (ongoing)' if active else ''))
        ids+=self._encode('.')+[self.tokenizer.eos_token_id]
        if not self.complete(ids): raise ValueError('reference grammar encoding failed')
        return ids


@torch.no_grad()
def generate(model, prefix, prompt_ids, tokenizer, grammar, max_new_tokens=512):
    backend=model.backend; prompt=backend.embed(prompt_ids)
    hidden=torch.cat((prefix.to(prompt),prompt),1)
    backend._check(hidden.shape[1]+max_new_tokens)
    sequences=backend.lm.generate(inputs_embeds=hidden,
        input_ids=torch.empty((1,0),device=backend.device,dtype=torch.long),
        attention_mask=torch.ones(hidden.shape[:2],device=backend.device,dtype=torch.long),
        do_sample=False,max_new_tokens=max_new_tokens,use_cache=True,
        prefix_allowed_tokens_fn=lambda batch,ids:grammar.allowed(ids.tolist()),
        pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    ids=sequences[0].tolist()
    return dict(text=tokenizer.decode(ids,skip_special_tokens=True),token_ids=ids,
                truncated=not grammar.complete(ids))
