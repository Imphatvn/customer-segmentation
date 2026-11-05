"""Basket validation and coverage for segment-level association analysis."""
import ast
from collections import Counter
import pandas as pd


def parse_basket(value):
    """Return normalized unique items and a parse status without guessing.

    Accept actual lists or stringified Python lists. Malformed baskets are
    reported separately rather than converted into fictitious product names.
    """
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [], 'invalid_syntax'
    if not isinstance(value, list):
        return [], 'not_a_list'
    if not all(isinstance(item, str) for item in value):
        return [], 'non_string_item'
    items = [' '.join(item.strip().lower().split()) for item in value]
    items = list(dict.fromkeys(item for item in items if item))
    return (items, 'valid') if items else ([], 'empty')


def prepare_baskets(baskets):
    """Keep every input row and attach parsing and original-list-length diagnostics."""
    result = baskets.copy()
    parsed = result['list_of_goods'].apply(parse_basket)
    result['items'] = parsed.map(lambda pair: pair[0])
    result['parse_status'] = parsed.map(lambda pair: pair[1])
    result['basket_size'] = result['items'].str.len()
    return result


def attach_assignments(baskets, register):
    """Left join transactions to a unique customer register and retain coverage states."""
    if register['customer_id'].isna().any() or not register['customer_id'].is_unique:
        raise ValueError('The assignment register needs unique, non-missing customer IDs.')
    result = baskets.merge(
        register[['customer_id', 'cluster', 'segment_name', 'assignment_status']],
        on='customer_id', how='left', validate='many_to_one', indicator=True,
    )
    result['coverage_status'] = 'pending_or_unassigned'
    result.loc[result['_merge'].eq('left_only'), 'coverage_status'] = 'customer_not_in_register'
    assigned = result['_merge'].eq('both') & result['cluster'].notna()
    result.loc[assigned, 'coverage_status'] = 'assigned'
    result.loc[result['customer_id'].isna(), 'coverage_status'] = 'missing_customer_id'
    return result.drop(columns='_merge')


def summarize_coverage(data):
    """Basket percentages use all input rows; distinct-customer counts exclude null IDs."""
    table = data.groupby('coverage_status', dropna=False).agg(
        baskets=('coverage_status', 'size'),
        distinct_customers=('customer_id', 'nunique'),
    )
    table['basket_pct'] = table['baskets'] / len(data) * 100
    return table


def product_catalog(data):
    """Count basket presence, not units; input must contain unique items per basket."""
    counts = Counter(item for basket in data['items'] for item in basket)
    table = pd.DataFrame(counts.items(), columns=['product_name', 'basket_count'])
    table['basket_pct'] = table['basket_count'] / len(data) * 100
    return table.sort_values(['basket_count', 'product_name'], ascending=[False, True]).reset_index(drop=True)


def segment_basket_summary(data):
    """Summarize eligible transaction rows, distinct customers, and basket sizes."""
    summary = data.groupby(['cluster', 'segment_name']).agg(
        baskets=('invoice_id', 'size'), customers=('customer_id', 'nunique'),
        mean_basket_size=('basket_size', 'mean'), median_basket_size=('basket_size', 'median'),
    )
    summary['baskets_per_customer'] = summary['baskets'] / summary['customers']
    return summary
