"""Supplementary instance-length attribution and explicitly non-temporal text review."""
import argparse
from collections import Counter
import json
from pathlib import Path


def bucket(duration):
    return '<1' if duration < 1 else '1-2' if duration < 2 else '2-4' if duration < 4 else '>=4'


def summarize(rows):
    return dict(instances=len(rows),missed_weight=sum(row['missed_weight'] for row in rows),
                miss_rate=sum(row['missed_weight'] for row in rows)/len(rows) if rows else None,
                ambiguous_instances=sum(row['identity_ambiguous'] for row in rows))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',required=True)
    args=parser.parse_args()
    root=Path(args.results)
    misses=json.loads((root/'miss-attribution.json').read_text())
    result={}
    review=[]
    order_audit={}
    controls=json.loads((root/'controlled-perturbations.json').read_text())
    notes={
        'validation':[
            '三个参考动作均未覆盖，生成另两个类别；无法由文本确定具体替代时段。',
            '保留翻身和下床，未覆盖开头的躺床类别。',
            '上床和翻身均未覆盖，输出走向椅子及坐下站起序列。',
            '两个躺床实例均未覆盖，保留翻身，新增下床类别。',
            '两次翻身只输出一次，不能确定缺少开头还是结尾实例。',
            '走向床和上床均未覆盖，输出椅子相关序列。',
            '三个床相关参考动作均未覆盖，输出椅子和走动序列。',
            '保留走向椅子和站起，未覆盖中间坐下；还生成参考标签未支持的站在桌前，词表外内容未进入额外类别计数。'],
        'test':[
            '两个参考动作均未覆盖，输出椅子站起及床相关序列。',
            '保留下床，躺床和走向房间未覆盖，额外输出坐在床上。',
            '保留前两个动作，未覆盖末尾坐到床上。',
            '保留走向床，上床和坐在床上未覆盖；新增床上坐下和站起。',
            '保留跌倒和从地面站起，走向床被另一个走动类别取代的文本候选。',
            '保留前两个动作，未覆盖末尾上床；还生成参考标签未支持的滚下楼梯，词表外内容未进入额外类别计数。',
            '床上的坐下站起被椅子坐下站起替代的文本候选，走向房间保留。',
            '参考三个类别均未覆盖，生成床上翻身、坐着、下床。']}
    for split in ('validation','test'):
        rows=json.loads((root/f'miss-instances-{split}.json').read_text())
        result[split]=dict(total=summarize(rows),
                           full_labeled_run_duration={name:summarize([row for row in rows if bucket(row['full_duration'])==name]) for name in ('<1','1-2','2-4','>=4')},
                           visible_duration_by_truncation={name:{cut:summarize([row for row in rows if bucket(row['duration'])==name and row['truncation']==cut]) for cut in ('complete','left','right','both')} for name in ('<1','1-2','2-4','>=4')},
                           caveat='Counts are labeled instances in overlapping windows and duplicate sensor views, not independent physical events. Duration follows speech-command label policy.')
        for row,note in zip(misses[split]['manual_review_queue'][:8],notes[split]):
            reference=Counter(row['reference_actions']);prediction=Counter(row['predicted_actions'])
            review.append(dict(row,split=split,review_status='assistant_text_audit',
                               missing_class_occurrences=list((reference-prediction).elements()),
                               extra_class_occurrences=list((prediction-reference).elements()),
                               note=note,physical_alignment_review='pending_human',
                               caveat='Text-only assistant review, not human radar annotation or reliable temporal substitution alignment. Extra class counts exclude out-of-vocabulary claims and do not measure all hallucinations.'))
        windows=json.loads((root/f'caption-windows-{split}.json').read_text())
        by_id={row['sample_id']:row for row in windows}
        for row in controls[split]['results']:
            if 'wrong_sample_id' in row:
                first,second=by_id[row['sample_id']],by_id[row['wrong_sample_id']]
                assert first['sensor']==second['sensor'] and first['actions']!=second['actions']
                assert abs(first['duration']-second['duration'])<=first['period']+1e-8
        # Correct terminal-frame-only view duplication; recorded predictions stay immutable.
        seen=set();unique=[]
        for i,row in enumerate(windows):
            key=(row['subject'],row['session'],row['start'])
            if key not in seen:seen.add(key);unique.append(i)
        pairs=[]
        for offset,i in enumerate(unique):
            a=windows[i]
            for j in unique[offset+1:]:
                b=windows[j]
                if a['sensor']!=b['sensor'] or a['actions']==b['actions'] or Counter(a['actions'])!=Counter(b['actions']) or len(set(a['actions']))<2:continue
                if a['recording']==b['recording'] and a['start']<b['stop'] and b['start']<a['stop']:continue
                pairs.append((a['sample_id'],b['sample_id']))
        original=controls[split]['real_order'];selected=original['results']
        assert len(pairs)==original['eligible_nonoverlapping_pairs']
        assert all((row['first_id'],row['second_id']) in pairs for row in selected)
        bag_correct=sum(Counter(row['first_score']['predicted_actions'])==Counter(by_id[row['first_id']]['actions'])
                        and Counter(row['second_score']['predicted_actions'])==Counter(by_id[row['second_id']]['actions']) for row in selected)
        order_audit[split]=dict(raw_physical_window_keys=original['physical_windows'],corrected_physical_window_keys=len(unique),
                                eligible_pairs=len(pairs),evaluated_pairs=len(selected),
                                unique_evaluated_windows=len({row[key] for row in selected for key in ('first_id','second_id')}),
                                both_action_multisets_correct=bag_correct,
                                pair_action_counts=dict(Counter(len(by_id[row['first_id']]['actions']) for row in selected)),
                                evaluated_predictions_unchanged=True)
    (root/'instance-extended-audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (root/'text-sample-review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
    (root/'real-order-dedup-audit.json').write_text(json.dumps(order_audit,indent=2),encoding='utf-8')
    print(json.dumps(dict(order_audit=order_audit,misses={key:value['total'] for key,value in result.items()}),indent=2))


if __name__=='__main__':
    main()
