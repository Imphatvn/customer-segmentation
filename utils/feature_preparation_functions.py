"""Feature preparation for the customer-segmentation project.

Adapted from the academic group's correlation and PCA workflow.
All transformations are fitted on the explicitly selected fitting population.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def summarize_features(data):
    """Summarize ranges, skew, and constant features before transformation."""
    summary = data.describe(percentiles=[0.5, 0.95, 0.99]).T
    summary['skewness'] = data.skew()
    summary['unique_values'] = data.nunique()
    return summary


def prepare_representations(data, spend_columns, variance_threshold=0.90):
    """Fit original-scaled, log-scaled, and log-scaled PCA candidates.

    data must have customer_id as its index and contain only selected numeric
    features. Only spending columns receive log1p; promotion share is unchanged
    before standardization. Returns matrices and reusable fitted artifacts.
    """
    if data.empty or not data.index.is_unique or data.index.hasnans:
        raise ValueError('Provide non-empty data with unique, non-missing customer IDs.')
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError('Selected features must be finite and complete.')
    if data[spend_columns].lt(0).any().any():
        raise ValueError('Spending must be non-negative for this workflow.')
    if data.nunique().le(1).any():
        raise ValueError('Remove constant features explicitly before preparation.')
    if not 0 < variance_threshold < 1:
        raise ValueError('variance_threshold must be between zero and one.')

    original_scaler = StandardScaler()
    original_scaled = pd.DataFrame(
        original_scaler.fit_transform(data), index=data.index, columns=data.columns
    )
    log_data = data.copy()
    log_data[spend_columns] = np.log1p(log_data[spend_columns])
    log_scaler = StandardScaler()
    log_scaled = pd.DataFrame(
        log_scaler.fit_transform(log_data), index=data.index, columns=data.columns
    )
    pca = PCA(n_components=variance_threshold, svd_solver='full', whiten=False)
    scores = pca.fit_transform(log_scaled)
    pca_data = pd.DataFrame(
        scores, index=data.index,
        columns=[f'PC{i + 1}' for i in range(scores.shape[1])]
    )
    representations = {
        'original_scaled': original_scaled,
        'log_scaled': log_scaled,
        'log_pca': pca_data,
    }
    artifacts = {
        'feature_columns': data.columns.tolist(),
        'spend_columns': list(spend_columns),
        'original_scaler': original_scaler,
        'log_scaler': log_scaler,
        'pca': pca,
    }
    return representations, artifacts


def transform_customers(data, artifacts):
    """Apply fitted transformations to customers with valid input features.

    This does not resolve questionable annual-spending estimates or assign
    clusters. Review eligibility before calling it.
    """
    selected = data[artifacts['feature_columns']].copy()
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError('New customers must have finite selected features.')
    spending = artifacts['spend_columns']
    if selected[spending].lt(0).any().any():
        raise ValueError('Spending must be non-negative.')
    original = pd.DataFrame(
        artifacts['original_scaler'].transform(selected),
        columns=selected.columns, index=selected.index
    )
    selected[spending] = np.log1p(selected[spending])
    logged = pd.DataFrame(
        artifacts['log_scaler'].transform(selected),
        columns=selected.columns, index=selected.index
    )
    scores = artifacts['pca'].transform(logged)
    compressed = pd.DataFrame(
        scores, index=selected.index,
        columns=[f'PC{i + 1}' for i in range(scores.shape[1])]
    )
    return {'original_scaled': original, 'log_scaled': logged, 'log_pca': compressed}


def plot_spending_transformation(data, columns):
    """Compare original and log spending on separate, clearly labeled axes."""
    fig, axes = plt.subplots(len(columns), 2, figsize=(12, 3 * len(columns)), squeeze=False)
    for row, column in enumerate(columns):
        label = column.replace('spend_per_year_', '').replace('_', ' ').title()
        axes[row, 0].hist(data[column], bins=50, color='#4477AA', alpha=0.85)
        axes[row, 1].hist(np.log1p(data[column]), bins=50, color='#228833', alpha=0.85)
        axes[row, 0].set_title(f'{label}: annual spending')
        axes[row, 1].set_title(f'{label}: log1p annual spending')
        for ax in axes[row]:
            ax.set_ylabel('Customers')
            ax.spines[['top', 'right']].set_visible(False)
        axes[row, 0].set_xlabel('Spending units per year')
        axes[row, 1].set_xlabel('log(1 + annual spending)')
    fig.tight_layout()
    return fig


def plot_correlations(data, method='spearman'):
    """Plot a readable correlation matrix for selected model features."""
    corr = data.corr(method=method)
    labels = [c.replace('spend_per_year_', '').replace('pct_prod_in_promotion', 'Promotion share')
              .replace('_', ' ').title() for c in corr.columns]
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(value) > 0.65 else 'black')
    ax.set_title(f'{method.title()} correlations — fitting population', pad=15)
    fig.colorbar(image, ax=ax, shrink=0.8, label='Correlation')
    fig.tight_layout()
    return fig


def pca_diagnostics(log_scaled, pca):
    """Return full-spectrum variance and retained-component coefficients.

    Coefficients are PCA directions, not feature-score correlations.
    """
    full_pca = PCA(svd_solver='full').fit(log_scaled)
    variance = pd.DataFrame({
        'component': np.arange(1, len(full_pca.explained_variance_ratio_) + 1),
        'explained_variance': full_pca.explained_variance_ratio_,
        'cumulative_variance': full_pca.explained_variance_ratio_.cumsum(),
    })
    coefficients = pd.DataFrame(
        pca.components_.T, index=log_scaled.columns,
        columns=[f'PC{i + 1}' for i in range(pca.n_components_)]
    )
    return variance, coefficients


def plot_pca_variance(variance, retained_components, threshold):
    """Show all components and the retained variance threshold."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(variance['component'], variance['explained_variance'],
           color='#4477AA', alpha=0.6, label='Individual variance')
    ax.plot(variance['component'], variance['cumulative_variance'],
            marker='o', color='#228833', label='Cumulative variance')
    ax.axhline(threshold, color='#CC6677', linestyle='--', label=f'Target: {threshold:.0%}')
    ax.axvline(retained_components, color='gray', linestyle=':',
               label=f'Retained: {retained_components}')
    ax.set(xlabel='Principal component', ylabel='Explained variance fraction',
           title='PCA variance — standardized log-spending representation', ylim=(0, 1.05))
    ax.set_xticks(variance['component'])
    ax.legend(loc='best')
    fig.tight_layout()
    return fig
