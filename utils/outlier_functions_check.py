import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors


def calculate_iqr_bounds(df, columns, multiplier=10):
    """Calculate bounds without modifying the customer data."""
    q1 = df[columns].quantile(0.25)
    q3 = df[columns].quantile(0.75)
    iqr = q3 - q1
    return pd.DataFrame({"q1": q1, "q3": q3, "iqr": iqr,
                         "lower_bound": q1 - multiplier * iqr,
                         "upper_bound": q3 + multiplier * iqr})


def flag_iqr_outliers(df, bounds):
    """Return a boolean flag for each customer and spending category."""
    data = df[bounds.index]
    return data.lt(bounds["lower_bound"]) | data.gt(bounds["upper_bound"])


def plot_spending_distribution(df, column, bounds):
    """Show a distribution summary and the actual screening thresholds."""
    data = df[column].dropna()
    title = column.replace("spend_per_year_", "").replace("_", " ").title()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.25, 0.75], vertical_spacing=0.12,
                        subplot_titles=["Box plot: standard whiskers", "Histogram: custom IQR screening bounds"])
    fig.add_trace(go.Box(x=data, boxpoints=False, marker_color="#4C78A8"), row=1, col=1)
    fig.add_trace(go.Histogram(x=data, nbinsx=60, marker_color="#4C78A8"), row=2, col=1)
    for key, label in [("lower_bound", "Lower bound"), ("upper_bound", "Upper bound")]:
        value = bounds.loc[column, key]
        if value >= data.min():
            fig.add_vline(x=value, line_color="#E45756", line_dash="dash",
                          annotation_text=label, row=2, col=1)
    fig.update_layout(title=f"Annual Spending — {title}", template="plotly_white",
                      height=550, showlegend=False, margin=dict(t=100))
    fig.update_xaxes(title_text="Annual spending", tickformat=",.0f", row=2, col=1)
    fig.update_yaxes(title_text="Customers", row=2, col=1)
    return fig


def calculate_neighbor_distances(X, min_samples=10):
    """Sorted distance to the min_samples-th neighbour, counting self."""
    model = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = model.kneighbors(X)
    return np.sort(distances[:, -1])


def evaluate_dbscan(X, eps_values, min_samples=10):
    """Compare cluster counts and noise rates; no preferred noise quota."""
    results = []
    for eps in eps_values:
        labels = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(X)
        results.append({"eps": float(eps), "min_samples": min_samples,
                        "clusters": len(set(labels) - {-1}),
                        "customers_flagged": int((labels == -1).sum()),
                        "percentage_flagged": float((labels == -1).mean() * 100)})
    return pd.DataFrame(results)


def detect_kmeans_outliers(X, n_clusters=70, percentile=97, random_state=42):
    """Flag large distances to the nearest prototype; return scores and model."""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    model.fit(X)
    scores = model.transform(X).min(axis=1)
    threshold = np.percentile(scores, percentile)
    return scores, scores > threshold, threshold, model


def detect_som_outliers(X, size=10, iterations=10000, percentile=97, random_state=42):
    """Flag large distances to the best-matching SOM unit."""
    from minisom import MiniSom
    values = np.asarray(X)
    som = MiniSom(size, size, values.shape[1], sigma=1.0,
                  learning_rate=0.5, random_seed=random_state)
    som.random_weights_init(values)
    som.train_random(values, num_iteration=iterations)
    weights = som.get_weights()
    scores = np.array([np.linalg.norm(row - weights[som.winner(row)]) for row in values])
    threshold = np.percentile(scores, percentile)
    return scores, scores > threshold, threshold, som


def summarize_flags(flags, columns):
    """Count customers, not repeated feature-level flags."""
    return pd.DataFrame({"customers_flagged": flags[columns].sum(),
                         "percentage_flagged": flags[columns].mean() * 100})


def plot_outlier_map(embedding, flags, title):
    """Plot all methods on the same diagnostic embedding without mutating data."""
    fig = go.Figure()
    flags = np.asarray(flags, dtype=bool)
    for value, label, color in [(False, "Not flagged", "#AAB7C4"), (True, "Flagged", "#E45756")]:
        subset = embedding[flags == value]
        fig.add_trace(go.Scattergl(x=subset[:, 0], y=subset[:, 1], mode="markers",
                                  name=label, marker=dict(color=color, size=4, opacity=0.55)))
    fig.update_layout(title=title, template="plotly_white", height=500,
                      xaxis_title="Diagnostic PC1", yaxis_title="Diagnostic PC2")
    return fig
