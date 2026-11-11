import torch


def test_streaming_chunks_keep_original_hidden_and_classifier():
    from continuous_mmwave.harrad.model import SegmentClassifier
    from continuous_mmwave.harrad.streaming import FrozenRDStream
    torch.set_num_threads(1); torch.manual_seed(12)
    original=SegmentClassifier('rd',0).eval(); stream=FrozenRDStream(original)
    x=torch.randn(2,11,40,64); mask=torch.ones(2,11,dtype=torch.bool); mask[0,8:]=False
    full=stream.forward_chunk(x,mask)
    state=None; parts=[]
    for s,e in [(0,3),(3,4),(4,9),(9,11)]:
        out=stream.forward_chunk(x[:,s:e],mask[:,s:e],state); state=out['state']; parts.append(out['hidden'])
    hidden=torch.cat(parts,1)
    torch.testing.assert_close(hidden,full['hidden'],atol=2e-6,rtol=1e-5)
    logits=original.classifier(hidden.sum(1)/mask.sum(1,keepdim=True))
    torch.testing.assert_close(logits,original(x,mask),atol=2e-6,rtol=1e-5)
    assert full['cells'].shape==(2,11,4,64)
    assert not any(p.requires_grad for p in stream.parameters())
    changed=x.clone(); changed[:,5:]*=100
    torch.testing.assert_close(stream.forward_chunk(changed,mask)['hidden'][:,:5],full['hidden'][:,:5])


def test_pair_buffer_keeps_odd_frame_and_only_flushes_explicitly():
    from continuous_mmwave.harrad.streaming import PairBuffer
    x=torch.arange(7*4*64).reshape(7,4,64).float()
    buffer=PairBuffer(); pieces=[]
    for s,e in [(0,1),(1,4),(4,5),(5,7)]: pieces.extend(buffer.push(x[s:e]))
    assert len(pieces)==3 and buffer.pending.shape==(1,4,64)
    pieces.extend(buffer.finish())
    assert len(pieces)==4 and pieces[-1][1].tolist()==[True,False]
    recovered=torch.cat([values[valid] for values,valid in pieces])
    torch.testing.assert_close(recovered,x)
