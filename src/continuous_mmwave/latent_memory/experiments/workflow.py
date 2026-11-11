"""Pre-registered diagnostic gate; no outcome-dependent budget extension."""

def oracle_pass(rows):
    return bool(rows) and all(r['metrics']['sequence_em']==1. and r['metrics']['state_em']==1. for r in rows)


def pilot_gate(oracle_exact, correct, null, swapped, expected_cases, actual_cases):
    checks=dict(oracle_exact=bool(oracle_exact),complete_panel=actual_cases==expected_cases,
                fit_sequence_exact=correct['sequence_em']==1.,
                fit_state_exact=correct['state_em']==1.,
                radar_beats_null=correct['edit_score']>null['edit_score'],
                radar_beats_swap=correct['edit_score']>swapped['edit_score'])
    passed=all(checks.values())
    return dict(passed=passed,g2_allowed=passed,checks=checks,
        interpretation='training-panel interface diagnostic only; not a generalization claim',
        next_action='lock formal minimum-useful-effect protocol before main experiment' if passed else 'stop at G0; preserve evidence; do not extend budget or run G2')
