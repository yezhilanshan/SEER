"""Declared G0/G2 scores; malformed cases remain in every denominator."""
from collections import Counter,defaultdict
from ..evaluation import parse_states
from ..data import PHRASES


def score(reference,text,truncated=False):
    prediction,states=parse_states(text); valid=prediction is not None
    guess=prediction if valid else []; truth=reference['sequence']
    counted=list(guess)
    if truncated and not valid:
        partial=' '.join(text.lower().split())
        if partial.startswith('the person '):
            for part in partial[len('the person '):].rstrip('.').split(', then '):
                phrase=part.removesuffix(' (ongoing)')
                if phrase not in PHRASES: break
                counted.append(PHRASES.index(phrase))
    table=list(range(len(guess)+1))
    for i,a in enumerate(truth,1):
        row=[i]
        for j,b in enumerate(guess,1): row.append(min(row[-1]+1,table[j]+1,table[j-1]+int(a!=b)))
        table=row
    tc=Counter(truth); pc=Counter(counted)
    return dict(parseable=valid,truncated=truncated,sequence_em=float(valid and not truncated and truth==guess),
                edit_score=1-table[-1]/max(len(truth),len(guess),1) if valid and not truncated else 0.,
                total_count_mae=abs(len(truth)-len(counted)),class_count_l1=sum(abs(tc[i]-pc[i]) for i in range(12)),
                state_em=float(valid and not truncated and truth==guess and reference['ongoing']==states),
                predicted_sequence=prediction,predicted_ongoing=states,counted_sequence=counted)


def aggregate(rows):
    fields=('edit_score','sequence_em','state_em','total_count_mae','class_count_l1','parseable','truncated')
    grouped=defaultdict(list)
    for row in rows: grouped[row['file']].append(row['metrics'])
    per_record={name:{k:sum(float(v[k]) for v in group)/len(group) for k in fields} for name,group in grouped.items()}
    macro={k:sum(v[k] for v in per_record.values())/max(1,len(per_record)) for k in fields}
    return dict(windows=len(rows),recordings=len(grouped),record_macro=macro,per_record=per_record)
