"""Deterministic case selection with explicit prevalence and paired clean rows."""
from __future__ import annotations


def select_failures(rows: list[dict]) -> dict:
    decisions = [r for r in rows if r['condition'] == 'D']
    clean = {r['source_id']: r for r in decisions if r['corruption'] == 'clean'}
    categories = {
        'confident_wrong_accepted': [r for r in clean.values() if r['accepted'] and not r['correct']],
        'correct_but_rejected': [r for r in clean.values() if r['correct'] and not r['accepted']],
        'corruption_damage': [r for r in decisions if r['corruption'] != 'clean' and r['severity'] == 'severe'
                              and not r['correct'] and r['accepted']
                              and clean[r['source_id']]['correct'] and clean[r['source_id']]['accepted']],
    }
    cases, counts, missing = [], {}, []
    for category, candidates in categories.items():
        counts[category] = {'candidate_rows': len(candidates),
                            'unique_source_images': len({r['source_id'] for r in candidates})}
        if not candidates:
            missing.append(category)
            continue
        selected = min(candidates, key=lambda r: (-r['calibrated_confidence'], r['source_id'], r['corruption']))
        cases.append({'category': category, 'row': selected, 'clean': clean[selected['source_id']]})
    return {'cases': cases, 'category_counts': counts, 'missing_categories': missing,
            'selection': 'D only; highest calibrated confidence, then source ID and corruption lexicographically',
            'interpretation': 'Representative selected cases, not random samples or prevalence estimates.'}
