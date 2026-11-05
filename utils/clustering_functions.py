"""Controlled K-Means experiments for the retail segmentation project."""
from itertools import combinations
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score


def evaluate_kmeans(representations, k_values, seeds=(11, 42, 73), n_init=10,
                    silhouette_sample=2000, sample_seed=42):
    """Fit on all fitting customers; evaluate silhouette on one fixed sample.

    Stability here means initialization stability on fixed prepared data, not
    validation against truth or robustness to resampling/preprocessing.
    One minimum-inertia run per representation/k is retained as a candidate.
    """
    first = next(iter(representations.values()))
    for X in representations.values():
        if not X.index.equals(first.index):
            raise ValueError('Representations must have identical ordered customer IDs.')
        if not np.isfinite(X.to_numpy()).all():
            raise ValueError('Feature matrices must be finite.')
    positions = np.sort(np.random.default_rng(sample_seed).choice(
        len(first), size=min(silhouette_sample, len(first)), replace=False))
    rows, stability_rows, models, assignments = [], [], {}, {}
    for representation, X in representations.items():
        values = X.to_numpy()
        for k in k_values:
            run_labels, run_models, run_inertia = [], [], []
            for seed in seeds:
                start = perf_counter()
                model = KMeans(n_clusters=k, n_init=n_init, random_state=seed,
                               max_iter=300, algorithm='lloyd')
                labels = model.fit_predict(values)
                counts = pd.Series(labels).value_counts()
                sampled_labels = labels[positions]
                unique = np.unique(sampled_labels)
                valid = 1 < len(unique) < len(positions)
                rows.append({
                    'representation': representation, 'k': k, 'seed': seed,
                    'inertia': model.inertia_,
                    'silhouette_sample': silhouette_score(values[positions], sampled_labels) if valid else np.nan,
                    'sample_clusters_seen': len(unique),
                    'davies_bouldin': davies_bouldin_score(values, labels) if len(counts) > 1 else np.nan,
                    'clusters_found': len(counts),
                    'min_cluster_count': int(counts.min()),
                    'min_cluster_pct': counts.min() / len(X) * 100,
                    'max_cluster_pct': counts.max() / len(X) * 100,
                    'iterations': model.n_iter_, 'iteration_limit_reached': model.n_iter_ >= 300,
                    'seconds': perf_counter() - start,
                })
                run_labels.append(labels); run_models.append(model); run_inertia.append(model.inertia_)
            for a, b in combinations(range(len(seeds)), 2):
                stability_rows.append({'representation': representation, 'k': k,
                    'seed_a': seeds[a], 'seed_b': seeds[b],
                    'ari': adjusted_rand_score(run_labels[a], run_labels[b])})
            best = int(np.argmin(run_inertia))
            candidate_id = f'{representation}__k{k}'
            models[candidate_id] = run_models[best]
            assignments[candidate_id] = pd.Series(run_labels[best], index=X.index, name='cluster')
    runs = pd.DataFrame(rows)
    stability = pd.DataFrame(stability_rows)
    summary = runs.groupby(['representation', 'k']).agg(
        silhouette_mean=('silhouette_sample', 'mean'), silhouette_sd=('silhouette_sample', 'std'),
        inertia_mean=('inertia', 'mean'), db_mean=('davies_bouldin', 'mean'),
        min_cluster_pct=('min_cluster_pct', 'min'), max_cluster_pct=('max_cluster_pct', 'max'),
        clusters_found_min=('clusters_found', 'min'),
        any_iteration_limit=('iteration_limit_reached', 'max'),
        mean_seconds=('seconds', 'mean'),
    ).reset_index()
    if not stability.empty:
        ari = stability.groupby(['representation', 'k']).agg(
            seed_ari_mean=('ari', 'mean'), seed_ari_min=('ari', 'min')).reset_index()
        summary = summary.merge(ari, on=['representation', 'k'])
    summary['candidate_id'] = summary['representation'] + '__k' + summary['k'].astype(str)
    return runs, summary, stability, models, assignments, first.index[positions]


def plot_kmeans_search(summary):
    """Separate panels per feature space; no cross-space inertia ranking."""
    names = summary['representation'].unique()
    fig, axes = plt.subplots(len(names), 3, figsize=(14, 3.5 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        data = summary[summary['representation'].eq(name)].sort_values('k')
        for col, metric, title in [(0, 'inertia_mean', 'Inertia'),
                                   (1, 'silhouette_mean', 'Sample silhouette'),
                                   (2, 'seed_ari_min', 'Minimum seed-pair ARI')]:
            axes[row, col].plot(data['k'], data[metric], marker='o', color='#4477AA')
            axes[row, col].set(title=f'{name}: {title}', xlabel='Number of clusters')
            axes[row, col].set_xticks(data['k'])
            axes[row, col].grid(alpha=.2)
    fig.tight_layout()
    return fig


def subsample_stability(X, reference_model, repeats=5, fraction=.8, n_init=10):
    """Refit on subsets and predict all fixed customers with frozen preparation.

    ARI against the reference assesses sensitivity to customer subsampling,
    not complete end-to-end preprocessing or temporal stability.
    """
    values = X.to_numpy()
    reference = reference_model.predict(values)
    rows = []
    for seed in range(100, 100 + repeats):
        indices = np.random.default_rng(seed).choice(len(X), int(len(X)*fraction), replace=False)
        model = KMeans(n_clusters=reference_model.n_clusters, n_init=n_init,
                       random_state=seed, max_iter=300).fit(values[indices])
        rows.append({'seed': seed, 'training_fraction': fraction,
                     'ari_to_reference': adjusted_rand_score(reference, model.predict(values))})
    return pd.DataFrame(rows)
