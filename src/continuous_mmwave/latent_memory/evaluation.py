"""Sequence metrics for controlled weak descriptions, with explicit parser limitations."""
import re
from collections import Counter
from .data import PHRASES


def parse_description(text):
    return parse_states(text)[0]


def parse_states(text):
    text=' '.join(text.lower().split())
    if text=='no annotated action has been observed yet.': return [],[]
    if not text.startswith('the person ') or not text.endswith('.'): return None,None
    parts=re.split(r',\s*then\s+',text[len('the person '):-1])
    labels=[]; states=[]
    for part in parts:
        ongoing=part.endswith(' (ongoing)')
        phrase=part[:-len(' (ongoing)')] if ongoing else part
        if phrase not in PHRASES: return None,None
        labels.append(PHRASES.index(phrase)); states.append(ongoing)
    return labels,states


def state_metrics(reference, ongoing, text):
    labels,states=parse_states(text)
    matched=labels==list(reference)
    return dict(state_sequence_exact=matched and states==list(ongoing),
                ongoing_accuracy_on_exact_sequence=(sum(a==b for a,b in zip(states,ongoing))/max(1,len(ongoing))) if matched else None)


def sequence_metrics(reference, predicted):
    if predicted is None:
        return dict(parseable=False,exact=False,edit_distance=None,normalized_edit_distance=None,instance_count_mae=None)
    previous=list(range(len(predicted)+1))
    for i,a in enumerate(reference,1):
        current=[i]
        for j,b in enumerate(predicted,1): current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    distance=previous[-1]; truth=Counter(reference); guess=Counter(predicted)
    return dict(parseable=True,exact=list(reference)==list(predicted),edit_distance=distance,
                normalized_edit_distance=distance/max(1,len(reference)),
                instance_count_mae=sum(abs(truth[i]-guess[i]) for i in range(12))/12)


def aggregate(rows):
    metrics=[r['metrics'] for r in rows]; parsed=[m for m in metrics if m['parseable']]
    return dict(queries=len(rows),parse_rate=len(parsed)/max(1,len(rows)),
                exact_sequence_rate=sum(m['exact'] for m in metrics)/max(1,len(rows)),
                mean_normalized_edit_distance=None if not parsed else sum(m['normalized_edit_distance'] for m in parsed)/len(parsed),
                mean_instance_count_mae=None if not parsed else sum(m['instance_count_mae'] for m in parsed)/len(parsed),
                truncated=sum(r['truncated'] for r in rows),
                state_sequence_exact_rate=sum(r['state_metrics']['state_sequence_exact'] for r in rows)/max(1,len(rows)),
                scope='annotated instance sequence; strict canonical English grammar; unknown-interval behavior and free paraphrases require human review')
