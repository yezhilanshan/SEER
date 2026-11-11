"""Export tables and an oracle confusion figure from aggregate evaluation JSON."""
import argparse
import csv
import json
from pathlib import Path


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tables(out, stem, summary):
    groups = {'pooled':summary['pooled'], **summary['by_subject'], **summary['by_record']}
    write_csv(out/(stem+'-metrics.csv'),[dict(group=group,arm=arm,**row)
                                         for group,arms in groups.items() for arm,row in arms.items()])


def oracle_report(out, oracle):
    import matplotlib.pyplot as plt
    import numpy as np

    classes = oracle['classes']
    matrix = np.asarray(oracle['confusion'])
    with (out/'oracle-confusion.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['truth / predicted',*classes])
        writer.writerows([label,*matrix[i]] for i,label in enumerate(classes))
    write_csv(out/'oracle-per-class.csv', [dict(row,errors=json.dumps(row['errors'])) for row in oracle['per_class']])
    fractions = matrix / matrix.sum(axis=1,keepdims=True)
    fig,ax = plt.subplots(figsize=(10,8),layout='constrained')
    heatmap = ax.imshow(fractions,cmap='Blues',vmin=0,vmax=1)
    labels = [c.replace('_',' ') for c in classes]
    ax.set_xticks(range(len(labels)),labels=labels,rotation=52,ha='right')
    ax.set_yticks(range(len(labels)),labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,
                    color='white' if fractions[i,j]>.55 else '#152335' if matrix[i,j] else '#8491a0')
    ax.set(xlabel='Predicted action',ylabel='Annotated action',
           title=f"Subject 7: oracle-interval action classification ({oracle['correct']}/{oracle['total']})")
    fig.colorbar(heatmap,ax=ax,fraction=.045,label='Fraction of true class')
    fig.savefig(out/'oracle-confusion.png',dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    out = args.output
    validation = json.loads((out/'subject7.json').read_text())
    tables(out,'subject7',validation)
    oracle_report(out,validation['oracle'])
    final = out/'subject8-9.json'
    if final.is_file():
        tables(out,'subject8-9',json.loads(final.read_text()))


if __name__=='__main__':
    main()
