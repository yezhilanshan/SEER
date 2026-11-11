import torch
import pytest


def test_clock_windows_split_recordings_without_action_aligned_crops():
    from continuous_mmwave.latent_memory.experiments.episodes import make_panel,window_reference
    records=[]
    for subject in ['subject1','subject2','subject7']:
        for suffix in ['01','02']:
            records.append(dict(file=f'{subject}_{suffix}',subject=subject,role='validation' if subject=='subject7' else 'train',
                                frames=130,cells_path='unused',instances=[dict(start=10,stop=100,label=0)]))
    panel=make_panel(records,lengths=(4,),fps=15)
    assert len(panel['fit'])==4 and len(panel['dev'])==4 and len(panel['validation'])==4
    assert [(e['start'],e['stop']) for e in panel['fit'][:2]]==[(0,60),(60,120)]
    ref=window_reference(records[0]['instances'],60,120)
    assert ref['sequence']==[0] and ref['left_censored']==[True] and ref['ongoing']==[False]
    assert ref['known_frames']==40 and ref['unknown_frames']==20
    assert ref['intervals']==[[0,40]]


def test_raw_age_memory_preserves_every_frame_and_distinct_access_scales():
    from continuous_mmwave.latent_memory.experiments.episodes import raw_memory
    cells=torch.arange(240.).view(240,1,1).expand(-1,4,64).clone()
    memory=raw_memory(cells,fps=15)
    assert [len(level) for level in memory.levels]==[30,60,30]
    assert [len(memory.candidates(g)) for g in range(3)]==[30,90,120]
    recovered=torch.cat([x.values[x.mask] for x in memory.candidates(2)])
    torch.testing.assert_close(recovered,cells)
    cells.zero_(); assert float(memory.candidates(2)[-1].values[-1,0,0])==239


def test_pilot_selection_covers_classes_and_never_uses_dev_or_validation():
    from continuous_mmwave.latent_memory.experiments.episodes import choose_pilot
    examples=[dict(uid=f'fit{i}',cohort='fit',seconds=4*(1+i%3),reference={'sequence':[i%12,(i+1)%12], 'ongoing':[False,True]}) for i in range(50)]
    examples[0]['reference']['sequence']=[0,0]
    examples[1]['seconds']=examples[2]['seconds']=4
    examples[1]['reference']['sequence']=[0,1]
    examples[2]['reference']['sequence']=[1,0]
    a=choose_pilot(examples,32,42); b=choose_pilot(examples,32,42)
    assert [x['uid'] for x in a]==[x['uid'] for x in b]
    assert len({x['uid'] for x in a})==32
    assert {label for x in a for label in x['reference']['sequence']}==set(range(12))


def test_pilot_rejects_missing_diagnostic_coverage():
    from continuous_mmwave.latent_memory.experiments.episodes import choose_pilot
    examples=[dict(uid=str(i),cohort='fit',seconds=4,reference={'sequence':[i%12], 'ongoing':[False]}) for i in range(40)]
    with pytest.raises(ValueError,match='diagnostic'):
        choose_pilot(examples)
