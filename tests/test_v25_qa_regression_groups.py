def test_added_subject_cannot_hide_old_qa_regression():
    from continuous_mmwave.v25.evaluate import compare_qa
    def score(value):
        return dict(identify_macro_recall=value,presence_balanced_accuracy=value,
                    identify_classes={str(i):dict(recall=value) for i in range(14)})
    current=score(.9); old=score(.9)
    current['groups']={'dev_regression':score(.8)}
    old['groups']={'dev_regression':score(.9)}
    result=compare_qa(current,old)
    assert not result['passed']
