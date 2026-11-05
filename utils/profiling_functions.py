"""Original-unit profiles with explicit customer-ID alignment."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_cluster_profiles(customers, labels, spend_columns):
    """Return numeric summaries relative to the same fitting population.

    customers and labels use customer_id as index. Spending shares describe
    recorded category mix, not basket co-occurrence or causal preferences.
    """
    if not customers.index.is_unique or not labels.index.is_unique:
        raise ValueError('Customer IDs must be unique.')
    labels = labels.reindex(customers.index)
    if labels.isna().any():
        raise ValueError('Every profiling customer needs a label.')
    data = customers.copy()
    data['cluster'] = labels
    grouped = data.groupby('cluster')
    sizes = grouped.size().rename('customers').to_frame()
    sizes['customer_pct'] = sizes['customers'] / len(data) * 100
    means = grouped[spend_columns].mean()
    medians = grouped[spend_columns].median()
    relative = means.divide(data[spend_columns].mean().replace(0, np.nan))
    total = data[spend_columns].sum(axis=1)
    data['total_annual_spending'] = total
    category_shares = data[spend_columns].div(total.replace(0, np.nan), axis=0)
    mean_shares = category_shares.assign(cluster=labels).groupby('cluster').mean()
    summary_columns = [c for c in ['total_annual_spending', 'pct_prod_in_promotion',
        'age', 'child_home', 'customer_lifetime', 'distinct_stores_visited',
        'lifetime_total_distinct_products'] if c in data]
    overview = data.groupby('cluster')[summary_columns].agg(['mean', 'median'])
    binary = [c for c in ['loyalty_card', 'complained', 'degree', 'male'] if c in data]
    rates = data.groupby('cluster')[binary].mean() * 100
    time_mix = pd.crosstab(data['cluster'], data['typical_time'], normalize='index') * 100         if 'typical_time' in data else pd.DataFrame()
    spending_share = data.groupby('cluster')['total_annual_spending'].sum()
    sizes['share_of_recorded_annual_spending_pct'] = spending_share / spending_share.sum() * 100
    return {'sizes': sizes, 'spending_means': means, 'spending_medians': medians,
            'relative_spending_means': relative, 'mean_category_shares': mean_shares,
            'overview': overview, 'binary_rates_pct': rates, 'shopping_time_pct': time_mix}


def plot_cluster_sizes(sizes):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(sizes.index.astype(str), sizes['customers'], color='#4477AA')
    ax.bar_label(bars, labels=[f'{v:,}' for v in sizes['customers']], padding=3)
    ax.margins(y=.15)
    ax.set(xlabel='Cluster ID', ylabel='Customers', title='Cluster sizes — fitting population')
    fig.tight_layout()
    return fig


def plot_relative_profiles(relative):
    """Log2 ratios centered on overall fitting-population mean; zero guarded."""
    transformed = np.log2(relative.clip(lower=.125, upper=8))
    fig, ax = plt.subplots(figsize=(12, max(4, .6 * len(relative))))
    image = ax.imshow(transformed, cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
    names = [c.replace('spend_per_year_', '').replace('_', ' ').title() for c in relative.columns]
    ax.set_xticks(range(len(names)), names, rotation=40, ha='right')
    ax.set_yticks(range(len(relative)), relative.index.astype(str))
    for i in range(len(relative)):
        for j in range(len(names)):
            value = relative.iloc[i, j]
            ax.text(j, i, f'{value:.2f}x' if np.isfinite(value) else 'NA',
                    ha='center', va='center', fontsize=8)
    ax.set(title='Cluster mean / overall mean (color clipped to 0.125–8x)', ylabel='Cluster ID')
    fig.colorbar(image, ax=ax, label='log2 relative mean')
    fig.tight_layout()
    return fig
