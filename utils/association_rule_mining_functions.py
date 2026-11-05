"""Exact pair-rule discovery and descriptive validation for transaction baskets.

Only singleton -> singleton rules are generated. Direct pair counting avoids
long itemsets and is exact for this restricted task; this is not Apriori.
"""
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = ['baskets', 'antecedent_count', 'consequent_count', 'joint_count',
           'antecedent_support', 'consequent_support', 'support', 'confidence',
           'lift', 'leverage', 'joint_customers']


def count_pairs(data, basket_column):
    """Count presence once per basket; retain distinct supporting customer counts."""
    singles, pairs, customers = Counter(), Counter(), {}
    for customer, basket in zip(data['customer_id'], data[basket_column]):
        items = sorted(set(basket))
        singles.update(items)
        for pair in combinations(items, 2):
            pairs[pair] += 1
            customers.setdefault(pair, set()).add(customer)
    return {'n': len(data), 'singles': singles, 'pairs': pairs,
            'customers': {pair: len(ids) for pair, ids in customers.items()}}


def pair_metrics(counts, antecedent, consequent):
    """A -> B metrics; no antecedent occurrences means undefined confidence."""
    n = counts['n']
    a = counts['singles'][antecedent]
    b = counts['singles'][consequent]
    pair = tuple(sorted((antecedent, consequent)))
    joint = counts['pairs'][pair]
    pa, pb, support = (a/n, b/n, joint/n) if n else (np.nan, np.nan, np.nan)
    confidence = joint/a if a else np.nan
    lift = confidence/pb if n and b else np.nan
    return dict(zip(METRICS, [n, a, b, joint, pa, pb, support, confidence, lift,
                              support-pa*pb, counts['customers'].get(pair, 0)]))


def discover_pair_rules(counts, min_count=30, min_support=.005,
                        min_confidence=.20, min_lift=1.10):
    """Apply exploratory thresholds to discovery data only; keep both directions."""
    rows = []
    for (a, b), count in sorted(counts['pairs'].items()):
        if count < min_count or count/counts['n'] < min_support:
            continue
        for antecedent, consequent in [(a, b), (b, a)]:
            metrics = pair_metrics(counts, antecedent, consequent)
            if metrics['confidence'] >= min_confidence and metrics['lift'] >= min_lift:
                rows.append({'antecedent': antecedent, 'consequent': consequent, **metrics})
    result = pd.DataFrame(rows, columns=['antecedent', 'consequent'] + METRICS)
    return result.sort_values(['lift', 'joint_count', 'antecedent', 'consequent'],
                              ascending=[False, False, True, True]).reset_index(drop=True)


def attach_rule_metrics(rules, counts, prefix):
    """Score fixed rules without rediscovering or filtering the evaluation sample."""
    result = rules.copy()
    records = [pair_metrics(counts, a, b) for a, b in zip(rules['antecedent'], rules['consequent'])]
    values = pd.DataFrame(records, index=rules.index, columns=METRICS).add_prefix(prefix)
    return pd.concat([result, values], axis=1)


def mine_segment_pairs(data, basket_column, level, thresholds):
    """Discover per segment; counts computed for segments and rest without joins exploding rows."""
    rules, counts = [], {}
    for cluster, group in data.groupby('cluster'):
        counts[int(cluster)] = count_pairs(group, basket_column)
        found = discover_pair_rules(counts[int(cluster)], **thresholds)
        found.insert(0, 'cluster', int(cluster))
        found.insert(1, 'level', level)
        rules.append(found)
    return pd.concat(rules, ignore_index=True), counts


def compare_rules(discovered, holdout, discovery, basket_column):
    """Add holdout segment and discovery-rest metrics. No outcome claims."""
    pieces = []
    for cluster, group in discovered.groupby('cluster'):
        current = attach_rule_metrics(group, count_pairs(holdout[holdout['cluster'].eq(cluster)], basket_column), 'holdout_')
        current = attach_rule_metrics(current, count_pairs(discovery[discovery['cluster'].ne(cluster)], basket_column), 'rest_')
        current['confidence_difference_vs_rest'] = current['confidence'] - current['rest_confidence']
        current['lift_difference_vs_rest'] = current['lift'] - current['rest_lift']
        pieces.append(current)
    if not pieces:
        result = discovered.copy()
        for prefix in ['holdout_', 'rest_']:
            for metric in METRICS:
                result[prefix + metric] = pd.Series(dtype=float)
        result['confidence_difference_vs_rest'] = pd.Series(dtype=float)
        result['lift_difference_vs_rest'] = pd.Series(dtype=float)
        return result
    return pd.concat(pieces, ignore_index=True)


def plot_top_pair_rules(rules, cluster, top_n=10):
    """Show exploratory discovery lift with actual A -> B labels and support counts."""
    chosen = rules[rules['cluster'].eq(cluster)].sort_values('lift', ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(11, max(3, .45*len(chosen)+1.5)))
    if chosen.empty:
        ax.text(.5, .5, 'No rules pass discovery thresholds', ha='center', transform=ax.transAxes)
        ax.set_axis_off()
    else:
        labels = [f'{a} -> {b} (n={int(n)})' for a, b, n in
                  zip(chosen['antecedent'], chosen['consequent'], chosen['joint_count'])]
        ax.barh(labels[::-1], chosen['lift'].to_numpy()[::-1], color='#4477AA')
        ax.axvline(1, linestyle='--', color='gray')
        ax.set_xlabel('Discovery lift (not incremental campaign effect)')
    ax.set_title(f'Cluster {cluster}: discovery-ranked pair rules')
    fig.tight_layout()
    return fig
