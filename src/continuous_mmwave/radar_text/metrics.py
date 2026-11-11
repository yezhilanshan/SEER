"""Restricted-vocabulary proxy metrics; these do not measure fine-grained grounding."""

import re
from collections import Counter

from .data import caption_from_actions


def extract_actions(text, phrases):
    # Prefer longer overlapping phrases ('sits down on the bed' over 'sits down').
    matches = []
    for action, phrase in phrases.items():
        pattern = r"(?<!\w)" + re.escape(phrase.rstrip(".").lower()) + r"(?!\w)"
        matches.extend((m.start(), m.end(), action) for m in re.finditer(pattern, text.lower()))
    occupied, selected = [], []
    for start, end, action in sorted(matches, key=lambda m: (-(m[1] - m[0]), m[0])):
        if not any(start < right and end > left for left, right in occupied):
            selected.append((start, action))
            occupied.append((start, end))
    return [action for _, action in sorted(selected)]


def score_caption(text, reference, phrases):
    predicted = extract_actions(text, phrases)
    common = sum((Counter(predicted) & Counter(reference)).values())
    previous = [0] * (len(reference) + 1)
    for action in predicted:
        current = [0]
        for j, expected in enumerate(reference, 1):
            current.append(previous[j - 1] + 1 if action == expected else max(current[-1], previous[j]))
        previous = current
    return dict(
        predicted_actions=predicted,
        action_recall=common / max(1, len(reference)),
        action_precision=common / max(1, len(predicted)),
        sequence_exact=float(predicted == reference),
        order_lcs=previous[-1] / max(1, len(reference), len(predicted)),
        template_exact=float(text.strip().lower() == caption_from_actions(reference, phrases).lower()),
        unmatched=float(not predicted),
    )
